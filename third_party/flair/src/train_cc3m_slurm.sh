#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --job-name=train_cc3m_slurm
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition PARTITION_NAME

source flair_env/bin/activate
cd flair/src

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


srun env -u CUDA_VISIBLE_DEVICES python -u -m torchrun \
    --nnode=$SLURM_JOB_NUM_NODES --nproc_per_node=gpu --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_backend=c10d -m main \
    --logs-dir ./logs \
    --model ViT-B-16-FLAIR \
    --use-flair-loss \
    --add-mps-loss \
    --train-dataset-type webdataset  \
    --lr 5e-4 \
    --warmup 2000 \
    --epochs 32  \
    --caption-sampling-mode diverse_sampling \
    --num-sampled-captions 8 \
    --log-every-n-steps 100 \
    --train-data 'datasets/cc3m_recap/cc3m-train-{0000..0575}.tar' \
    --train-num-samples 2823019 \
    --delete-previous-checkpoint \
    --batch-size 128 \
    --precision amp \
    --workers 48 \
    --beta1 0.9 \
    --beta2 0.98 \
    --wd 0.5 \
    --eps 1e-8 \

