torchrun --nnode 1 --nproc_per_node 4 -m main \
    --model ViT-B-16-FLAIR \
    --train-data './datasets/cc3m_recaptioned/' \
    --train-num-samples  2823019 \
    --train-dataset-type webdataset  \
    --use-flair-loss \
    --add-mps-loss \
    --num-sampled-captions 8 \
    --log-every-n-steps 200 \
    --caption-sampling-mode diverse_sampling \
    --batch-size 128 \
    --precision amp \
    --workers 48 \
    --delete-previous-checkpoint \
    --beta1 0.9 \
    --beta2 0.98 \
    --wd 0.5 \
    --eps 1e-8 \
    


    



