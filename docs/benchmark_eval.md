# Benchmark Evaluation

This repo evaluates prepared image-text benchmarks from:

```bash
/vepfs/dataset/benchmarks
```

The launcher is:

```bash
PYTHONPATH=/vepfs/code/SPVD/src python -m eval.benchmarks
```

It wraps `python -m eval.benchmarks` and writes timestamped runs under:

```bash
/vepfs/code/SPVD/outputs/benchmark_eval
```

## Smoke Test

Run a minimal CPU smoke test:

```bash
cd /vepfs/code/SPVD
PYTHONPATH=/vepfs/code/SPVD/src python -m eval.benchmarks --dry_run --limit 2 --models clip_vit_b32_openai --device cpu --batch_size 2
```

This checks benchmark loading, local model loading, scoring, raw CSV writing, and summary generation.

## Full Run

Run the configured model set on GPU:

```bash
cd /vepfs/code/SPVD
PYTHONPATH=/vepfs/code/SPVD/src python -m eval.benchmarks
```

Use a model subset:

```bash
MODELS=clip_vit_b32_openai,openclip_vit_b32_laion2b PYTHONPATH=/vepfs/code/SPVD/src python -m eval.benchmarks
```

Use a cap for debugging:

```bash
LIMIT=128 MODELS=clip_vit_b32_openai PYTHONPATH=/vepfs/code/SPVD/src python -m eval.benchmarks
```

## SPVD Checkpoints

`zero_train_diagnostics.models.SPVDWrapper` supports text-cued SPVD scoring through the full paired forward path, so image features are computed with the paired text cue rather than cached as static image embeddings.

Enable this block in `configs/benchmark_eval.yaml` when a checkpoint exists:

```yaml
- name: spvd_soft_cue_vitb16
  model_type: spvd
  model_name: SPVD-ViT-B-16
  checkpoint_path: /vepfs/code/SPVD/outputs/spvd_soft_cue_vitb16_cc3m/checkpoints/epoch_0001.pt
  image_size: 224
```

## Current Coverage

The prepared launcher evaluates:

- ARO: VG-Attribution, VG-Relation, COCO/Flickr order negatives.
- SugarCrepe: replace/swap/add hard negatives.
- Winoground: image/text/group scores.

COCO/Flickr retrieval data is prepared under `/vepfs/dataset/benchmarks`; retrieval-specific multi-caption metrics should be handled by a dedicated retrieval evaluator rather than the pairwise compositional evaluator.
