# Data Preparation for Text-Image Retrieval

### Annotation Files
We pre-processed and unified the annotations for various datasets to be in `.json` format to standardize them. These annotation files are stored under `datasets/` directory of this repo. To use our inference code properly, you should also use the same annotation files, the detailed instructions are as follows:

### Datasets list:
- [MSCOCO](#coco)
- [FLICKR30K](#flickr)
- [DOCCI](#docci)
- [IIW](#IIW)
- [ShareGPT4v](#share4v)
- [DCI](#DCI)
- [Urban1k](#urban1k)


### <span id ='coco'> MSCOCO dataset
```
$coco/
|–– images/
|–––– val2017/
|–––––– 000000134722.jpg
|–––––– 000000177015.jpg
|–––––– ...
|–– annotations/
|–––– captions_val2017.json
```
Step 1. Download validation images from [COCO 2017 Val Images](https://cocodataset.org/#download), unzip them to `coco/images/val2017`

Step 2. Download the 2017 Val annotations, place it under `coco/annotations/captions_val2017.json`

### <span id ='coco'> FLCIKR30K dataset
```
$flickr30k-images/
|––  2217728745.jpg 
|––  2217728745.jpg
|––  ...
|––  flickr30k_val.json
|––  flickr30k_test.json
```
Step 1. Download  [flickr30k dataset](https://huggingface.co/datasets/nlphuji/flickr30k), unzip them under `flickr30k-images/`, all the images and annotations files will be structured as above

### <span id ='docci'> DOCCI dataset
```
$docci/
|––  images/
|––––  test_01427.jpg
|––––  test_01428.jpg
|––––  ...
|––  annotations/
|–––– test_annotations.json
```
Step 1. Download  [DOCCI Images](https://storage.googleapis.com/docci/data/docci_images.tar.gz), unzip them under `docci/images/`, note that we only need the 5K test images here.

Step 2. Directly copy the `test_annotations.json` in this repo and put it under `docci/annotations`. This annotation file documents the mapping between all test images with all fine-grained captions.

### <span id ='iiw'> IIW dataset

```
$imageinwords/
|–– dci/
|–– docci/
|–– docci_aar/
|–– finegrained_annotations.json
```

**Download human annotated data following [IIW](https://github.com/google/imageinwords/tree/main/datasets), including IIW-400, DCI-Test, DOCCI-Test**:

Step 1: Download [DCI](https://github.com/facebookresearch/DCI) to path_to_dci_dataset.

Step 2: Download DOCCI images and AAR images from [DOCCI](https://google.github.io/docci/#downloads) dataset. Unzip the files to path_to_docci_dataset/images and path_to_docci_dataset/images_aar, respectively.

Step 3: Directly copy `finegrained_annotations.json` in this repo and put it under `imageinwords\`.


### <span id ='share4v'> ShareGPT4v dataset

```
$share4v/
|–– sa_000000/
|–––– images/
|–––––– sa_1.jpg
|–––––– sa_2.jpg
|–––––– ...
|–– sa_000001/
|–– ...
```

Step 1. Download tar files from [SA-1B](https://huggingface.co/datasets/sailvideo/SA-1B) to `share4v/`.

Step 2. Unzip all tar files.

For the annotations, we have resaved the top 10k samples from [share-captioner_coco_lcs_sam_1246k_1107.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main) in dataloaders/share4v/share4v_sam_10k.json.


### <span id ='dci'> DCI dataset

```
$dci/
|–– densely_captioned_images/
|–––– annotations/
|–––– photos/
|–––– splits.json

```

**Download data following [DCI](https://github.com/facebookresearch/DCI)**:

Step 1. Download [dci.tar.gz](https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz) and unzip the file in `dci/densely_captioned_images`.

Step 2. Download the archive sa_000138.tar and extract the images to the `dci/densely_captioned_images/photos folder`.


### <span id ='urban1k'> Urban1k dataset
```
$Urban1k/
|––  images/
|––––  221.jpg
|––––  222.jpg
|––––  ...
|––  annotations/
|–––– annotations.json
```
Step 1. Download  [Urban1K](https://huggingface.co/datasets/BeichenZhang/Urban1k), unzip them, only put the images(without the caption folder)under `Urban1k/images/`.

Step 2. Directly copy the `annotations.json` in this repo and put it under `Urban1k/annotations`. This annotation file documents the mapping between each image with its corresponding long caption.
