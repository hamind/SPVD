# CC3M Dataset Inspection

## 1. Dataset Root

- Root path: `/vepfs/dataset/cc3m`
- Inspection date: 2026-05-09
- Host: `volc_dev`

## 2. Directory Structure

`find . -maxdepth 2 -type d | head -100` returned only:

```text
.
```

The dataset root is a flat shard directory. No first-level or second-level subdirectories were detected.

Representative root files:

```text
./cc3m-train-0000.tar
./cc3m-train-0001.tar
...
./cc3m-train-0574.tar
./cc3m-train-0575.tar
```

The root directory size is approximately `261G`.

## 3. File Type Statistics

Root-level file counts:

| Type | Count |
| --- | ---: |
| `.csv` | 0 |
| `.tsv` | 0 |
| `.json` | 0 |
| `.jsonl` | 0 |
| `.parquet` | 0 |
| `.tar` | 576 |
| `.jpg` | 0 |
| `.jpeg` | 0 |
| `.png` | 0 |
| `.webp` | 0 |
| `.txt` | 0 |

Split-name counts at the root:

| Pattern | Count |
| --- | ---: |
| `train` | 576 |
| `val` | 0 |
| `valid` | 0 |
| `validation` | 0 |
| `test` | 0 |

## 4. Train / Val / Test Split

Only train shards were detected by filename:

```text
cc3m-train-0000.tar ... cc3m-train-0575.tar
```

No validation or test split was automatically found under `/vepfs/dataset/cc3m`.

## 5. Caption File Or Caption Field

Sampling `cc3m-train-0000.tar` showed WebDataset-style triples:

```text
000000000.jpg
000000000.json
000000000.txt
000000001.jpg
000000001.json
000000001.txt
```

The `.txt` member contains the raw caption:

```text
a river has burst it 's banks and has spread out onto arable farmland alongside
```

The `.json` member also contains a `caption` field plus metadata:

```json
{
  "caption": "a river has burst it 's banks and has spread out onto arable farmland alongside",
  "url": "https://ak7.picdn.net/shutterstock/videos/8592247/thumb/1.jpg",
  "key": "000000000",
  "status": "success",
  "error_message": null,
  "width": 852,
  "height": 480,
  "exif": "{}",
  "original_width": 852,
  "original_height": 480
}
```

The recommended default caption source is the `.txt` member, with `.json` retained as optional metadata.

## 6. Image Path Or Image Shard Format

Images are embedded in tar shards as `.jpg` members sharing the same basename as `.txt` and `.json` metadata files. This is WebDataset-style key grouping:

```text
<key>.jpg
<key>.txt
<key>.json
```

Example key:

```text
000000000
```

## 7. Recommended Data Reading Mode

Use the WebDataset-style reader:

```yaml
data:
  type: webdataset
  root: /vepfs/dataset/cc3m
  train_data: /vepfs/dataset/cc3m/cc3m-train-{0000..0575}.tar
  image_key: jpg
  caption_key: txt
  metadata_key: json
```

Do not assume `train.csv`, `val.csv`, `image_key`, or `caption_key` for a manifest layout because no manifest files were found.

## 8. Fields Requiring User Input

Because no validation split was detected, retrieval evaluation cannot run against CC3M until one of the following is provided:

- `data.val_data`: a WebDataset validation shard pattern.
- `data.val_file`: a manifest file with `image_key` and `caption_key`.
- A separate benchmark dataset config for retrieval evaluation.

Training can use the detected train shards. Evaluation configs intentionally keep validation data unset and will raise a clear error until the user fills the validation source.
