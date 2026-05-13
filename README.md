# SPVD

`SPVD` is an independent CLIP-style image-text pretraining codebase for SPVD experiments. The project keeps the structure compact: `python -m main` is the training entrypoint, and package-local modules under `src/` own params, factory, data, training, distributed utilities, logging, scheduling, checkpointing, evaluation, and SPVD model code.

OpenCLIP is used as the dependency for generic CLIP pieces such as ViT/Text backbones, tokenizers, and transforms. The project implements its own SPVD model, InfoNCE loss, and sigmoid loss while delegating ordinary CLIP model construction to OpenCLIP.

## Environment

Use the existing conda environment:

```bash
conda activate openclip
cd /vepfs/code/SPVD
export PYTHONPATH=/vepfs/code/SPVD/src${PYTHONPATH:+:${PYTHONPATH}}
```

Do not create a new conda environment for this project.

## Structure

```text
src/
  checkpoint.py
  clip_components.py
  data.py
  distributed.py
  factory.py
  logger.py
  losses.py
  main.py
  model.py
  params.py
  scheduler.py
  tokenizer.py
  training.py
  assets/
  eval/
  model_configs/
```

The training entrypoint is:

```bash
python -m main --config configs/train_cc3m_vitb16.yaml
```

## CC3M

The original dataset root is `/vepfs/dataset/cc3m`. The training configs now
use the rewritten long-caption dataset at `/vepfs/dataset/cc3m_longcaption`.

- Flat WebDataset-style train shards: `cc3m-train-0000.tar` through `cc3m-train-0575.tar`.
- No validation split was detected.
- Each sampled record is grouped as `<key>.jpg`, `<key>.txt`, and `<key>.json`.
- Rewritten shards store `longSV` in `.txt` and all caption variants under `.json["captions"]`.
- Long-caption relabels also have a SQLite index at `/vepfs/code/SPVD/cache/long_caption/cc3m_longSV_captions.sqlite` for on-the-fly experiments with the original shards.

The default train pattern is:

```text
/vepfs/dataset/cc3m_longcaption/cc3m-train-{0000..0575}.tar
```

Retrieval evaluation needs a validation shard pattern or manifest supplied in `configs/eval_retrieval.yaml`.

## Baselines

Run local CLIP or SigLIP-loss baselines directly through the Python entrypoint:

```bash
python -m main --config configs/train_cc3m_vitb16.yaml
python -m main --config configs/train_cc3m_vitb16.yaml --siglip --loss-name siglip
```

The optional baseline env files remain under `configs/baselines/` for reference, but the old launcher scripts were removed to keep the project clean.

Checkpoint policy: training writes `epoch_XXXX.pt` at the configured save frequency and always writes `epoch_final.pt` under the run's `checkpoints/` directory.

## DDP Data Loading

For the target runtime of 26 CPU cores, about 450GiB available memory, and two DDP ranks, the current CC3M WebDataset loader is tuned to:

```yaml
data:
  batch_size: 640
  num_workers: 13
  prefetch_factor: 2
  persistent_workers: true
  long_caption_file: null
  metadata_key: null
  pin_memory: true
  tar_mode: r|*
```

This uses 26 DataLoader workers across two ranks. `batch_size: 640` is per rank, so the two-GPU global batch is 1280. The loader uses streaming tar reads and persistent workers; the default path reads only `.jpg + .txt` from the rewritten shards.

## Tests

```bash
conda activate openclip
cd /vepfs/code/SPVD
pytest tests/
```

## Future SPVD Work

- Extend shared/private visual decomposition in `src/model.py`.
- Add a dedicated losses module when SPVD-specific objectives are implemented.
- Add ARO, SugarCrepe, and Winoground implementations under `src/eval/benchmarks.py`.
