# Benchmark Evaluation

This repo evaluates prepared image-text benchmarks from:

```bash
/vepfs/dataset/benchmark
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

Run multi-GPU evaluation with torchrun:

```bash
cd /vepfs/code/SPVD
PYTHONPATH=/vepfs/code/SPVD/src:/vepfs/code/SPVD \
torchrun --nproc_per_node=2 -m eval.benchmarks --config configs/benchmark_eval.yaml
```

Each rank evaluates `samples[rank::world_size]`, writes `raw_results/*_rank{rank}.jsonl`, and rank 0 merges the rank JSONL files back to the usual raw CSV schema. Pipeline knobs live in the benchmark config: `num_workers_per_gpu`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `batch_size`, `pair_chunk_size`, and `dtype`.

Use a model subset:

```bash
MODELS=clip_vit_b32_openai,openclip_vit_b32_laion2b PYTHONPATH=/vepfs/code/SPVD/src python -m eval.benchmarks
```

Use a cap for debugging:

```bash
LIMIT=128 MODELS=clip_vit_b32_openai PYTHONPATH=/vepfs/code/SPVD/src python -m eval.benchmarks
```

## SPVD Checkpoints

`zero_train_diagnostics.models.SPVDWrapper` supports exact text-conditioned scoring with `eval_mode.spvd_pairwise_mode: exact_cached`. It caches only image tokens and text cues, then runs soft-cue decomposition per image-caption pair:

```text
score(image, caption) = dot(
  soft_cue_decomposition(image_tokens, soft_cues(caption)).shared_visual_features,
  text_global(caption)
)
```

Do not cache final SPVD image features for pairwise evaluation, because they depend on the paired caption.

Retrieval supports `retrieval_mode: global` and `retrieval_mode: rerank_topk`. The rerank mode first uses global image/text features to select top-K candidates, then applies exact SPVD text-conditioned scoring only to those candidate pairs instead of scoring the full image-by-text matrix.

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

COCO/Flickr retrieval data is prepared under `/vepfs/dataset/benchmark`; retrieval-specific multi-caption metrics should be handled by a dedicated retrieval evaluator rather than the pairwise compositional evaluator.
