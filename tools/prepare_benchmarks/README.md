# SPVD Image-Text Benchmark Preparation

This directory prepares the first SPVD benchmark batch under:

```bash
/vepfs/dataset/benchmark
```

The goal is to evaluate whether visual shared/private decomposition preserves language-grounded shared semantics across retrieval, attributes, relations, word order, hard-negative caption discrimination, and compositional image-text matching.

## Quick Start

```bash
cd /vepfs/code/SPVD/tools/prepare_benchmarks

bash prepare_all.sh --p0
bash prepare_all.sh --p1
bash prepare_all.sh --all
bash prepare_all.sh --only coco
bash prepare_all.sh --only sugarcrepe

python3 verify_benchmarks.py
python3 build_manifest.py
```

Logs are written to:

```bash
/vepfs/dataset/benchmark/download_logs
```

Manifests and verification reports are written to:

```bash
/vepfs/dataset/benchmark/manifests
```

## Benchmarks And Sources

COCO Retrieval:

- Purpose: COCO image-to-text and text-to-image retrieval.
- Official source: https://cocodataset.org/#download
- Recommended download: official `train2014.zip`, `val2014.zip`, `val2017.zip`, `annotations_trainval2014.zip`, and `annotations_trainval2017.zip` from `images.cocodataset.org`; Karpathy caption split from `https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip`.
- Account/token: none for official COCO files.
- Terms: COCO annotations are Creative Commons Attribution 4.0; images keep Flickr image licenses and Flickr Terms of Use.
- Reuse policy: SugarCrepe/ARO should reference the COCO image roots here rather than storing duplicate images. Classic Karpathy retrieval uses COCO 2014 train/val splits; COCO 2017 val is also prepared because many modern image-text benchmark pipelines and SugarCrepe use COCO2017 val images.

Flickr30k Retrieval:

- Purpose: Flickr30k retrieval and ARO Flickr30k-Order.
- Official source: https://shannon.cs.illinois.edu/DenotationGraph/
- Recommended download: request access through the official UIUC form, then set `FLICKR30K_IMAGE_ROOT=/path/to/flickr30k-images` before running `download_flickr30k.sh`.
- Account/token: official access is form-gated/manual. Kaggle mirrors are not used by default; set `ALLOW_FLICKR30K_KAGGLE=1` only if you have accepted the relevant terms and configured Kaggle.
- Terms: captions are released under Creative Commons Attribution-ShareAlike; images are from Flickr, provided for non-commercial research/education, and must follow Flickr Terms of Use.

ARO:

- Purpose: VG-Attribution, VG-Relation, COCO-Order, Flickr30k-Order compositionality.
- Official source: https://github.com/mertyg/vision-language-models-are-bows
- Recommended download: clone the author repository; use its Google Drive IDs for VG attribution/relation annotations and `vgr_vga_images.zip`; use the official ARO storage URLs for COCO/Flickr order JSON files.
- Account/token: none if Google Drive/gdown works; Flickr30k images still require authorized local access.
- Terms: ARO code is MIT; underlying Visual Genome/GQA subset, COCO, and Flickr30k assets keep their original terms.
- Reuse policy: COCO/Flickr order image paths are symlinked to `benchmarks/coco` and `benchmarks/flickr30k`.

SugarCrepe:

- Purpose: hard-negative caption discrimination with replace/swap/add categories.
- Official source: https://github.com/RAIVNLab/sugar-crepe
- Recommended download: clone the author repository and copy `data/*.json`.
- Account/token: none for annotations.
- Terms: SugarCrepe work/code is MIT; examples are derived from public prior datasets, especially COCO, which keeps its own terms.
- Reuse policy: uses COCO 2017 val images via `benchmarks/coco/images/val2017`.

Winoground:

- Purpose: fine-grained visio-linguistic compositional reasoning over two images and two captions.
- Official source: https://huggingface.co/datasets/facebook/winoground
- Recommended download: accept Hugging Face gated terms, export `HF_TOKEN`, then run `python3 download_winoground.py`. The script uses direct Hugging Face file URLs for `data/examples.jsonl` and `data/images.zip`; it does not require `huggingface_hub`. Set `HF_ENDPOINT=https://hf-mirror.com` on networks where `huggingface.co` is blocked.
- Account/token: required unless an authorized local copy already exists.
- Terms: gated Hugging Face dataset; access requires sharing contact information and agreeing to research-only use. The full license agreement is in the dataset files.
- Failure behavior: if token/access is missing, status becomes `gated_or_auth_required` and P0 continues.

SVO-Probes:

- Purpose: subject/verb/object grounding and verb semantics.
- Official source: https://github.com/google-deepmind/svo_probes
- Recommended download: clone repository or download `svo_probes.csv` and `image_urls.txt`.
- Account/token: none.
- Terms: data is CC BY 4.0; code is Apache-2.0. Images are external URLs and keep upstream rights.

BiVLC:

- Purpose: bidirectional hard-negative image-text retrieval, complementing SugarCrepe with text-to-image checks.
- Official source: https://huggingface.co/datasets/imirandam/BiVLC and https://imirandam.github.io/BiVLC_project_page
- Recommended download: direct Hugging Face parquet URLs from `https://huggingface.co/api/datasets/imirandam/BiVLC/parquet`; the script downloads those URLs with Python stdlib and does not require `huggingface_hub`. It honors `HF_ENDPOINT` and falls back to `https://hf-mirror.com` when `huggingface.co` is unreachable.
- Account/token: usually not gated, but `HF_TOKEN` can be supplied for authenticated Hub access.
- Terms: dataset card states MIT License; COCO-derived and generated image assets should be used with upstream terms in mind.

## Expected Directory Layout

```bash
/vepfs/dataset/benchmark/
  coco/
    images/train2014/
    images/val2014/
    images/val2017/        # optional, reused by SugarCrepe
    annotations/
    splits/
    manifest.json
  flickr30k/
    images/
    annotations/
    splits/
    manifest.json
  aro/
    vg_attribution/
    vg_relation/
    coco_order/
    flickr30k_order/
    annotations/
    manifest.json
  sugarcrepe/
    annotations/
    splits/
    coco_val2017_images -> ../coco/images/val2017
    manifest.json
  winoground/
    images/
    examples.jsonl
    metadata/
    manifest.json
  svo_probes/
  bivlc/
  manifests/
  download_logs/
  _raw/
```

## Evaluation Path Usage

Evaluation code should prefer these roots:

```bash
COCO_ROOT=/vepfs/dataset/benchmark/coco
FLICKR30K_ROOT=/vepfs/dataset/benchmark/Flickr30k
ARO_ROOT=/vepfs/dataset/benchmark/aro
SUGARCREPE_ROOT=/vepfs/dataset/benchmark/sugarcrepe
WINOGROUND_ROOT=/vepfs/dataset/benchmark/winoground
```

Use `/vepfs/dataset/benchmark/manifests/benchmarks_manifest.json` as the machine-readable source of annotation files, split files, image roots, and status.

## Verification

```bash
python3 /vepfs/code/SPVD/tools/prepare_benchmarks/verify_benchmarks.py
```

The verifier checks directories, annotations, sample captions/positive-negative captions, and image paths. Gated or unauthorized datasets produce a clear status instead of failing the whole run.

## Sync

Dry run:

```bash
bash sync_benchmarks.sh --src /vepfs/dataset/benchmark --dst <TARGET_PATH> --dry-run
```

Actual rsync:

```bash
bash sync_benchmarks.sh --src /vepfs/dataset/benchmark --dst <TARGET_PATH>
```

For object storage or remote servers, fill in the target rsync address or adapt the command template. This script does not delete source or destination files by default.

## Duplicate Cleanup

Use this when `/vepfs/dataset/benchmark` and older dataset roots under `/vepfs/dataset` contain the same annotation files:

```bash
python3 dedupe_benchmarks.py
python3 dedupe_benchmarks.py --apply
python3 verify_benchmarks.py
python3 build_manifest.py
```

The cleanup only replaces byte-identical annotation copies with symlinks. It reports large raw archives, such as COCO zip files, but keeps them by default because those archives are useful for resumable downloads and rebuilding.

## Current Status

Run `python3 build_manifest.py` after preparation. The authoritative current state is:

```bash
/vepfs/dataset/benchmark/manifests/benchmarks_manifest.json
/vepfs/dataset/benchmark/manifests/verify_report.json
```

On `volc_dev`, pre-existing local copies were detected for ARO VG/COCO/Flickr order assets, Flickr30k images, SugarCrepe annotations/COCO2017 val images, and Winoground. COCO 2014 train images and official COCO caption annotations may still need to be downloaded if they are not already present when `download_coco.sh` runs.
