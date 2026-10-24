# Data, sources and terms

The gallery has 1,000 images: 500 from COCO val2017 and 500 from TextCaps validation. Every
human caption of an image is a query whose one relevant item is that image. This file records
exactly which files are used, the terms attached to them, what is committed and what is not.

Checked 2026-10-25 (live HTTP status in the tables).

## Verdict on the redistribution gate

Passed. Nothing in the published terms forbids what this repo commits (image ids, captions,
URLs, hashes, OCR text and derived embeddings). Images are never committed, so the per-image
Flickr and Open Images licences do not attach to anything in the repository. One caveat is
stated under "Residual risk".

## Sources

### 1. COCO 2017 (visual slice)

| Item | URL | Status | Size | Terms |
|---|---|---|---|---|
| Annotations zip (contains `annotations/captions_val2017.json`) | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` | 200 | 252,907,541 B | CC BY 4.0 |
| Images, fetched by file name | `http://images.cocodataset.org/val2017/<file_name>` (e.g. `000000000139.jpg`) | 200 | 80-200 kB each | Flickr Terms of Use, per image |
| Terms page | `https://cocodataset.org/#termsofuse` | 200 | | read live |

Terms as published: "The annotations in this dataset along with this website belong to the COCO
Consortium and are licensed under a Creative Commons Attribution 4.0 License." Images: "The COCO
Consortium does not own the copyright of the images. Use of the images must abide by the Flickr
Terms of Use. The users of the images accept full responsibility for the use of the dataset,
including but not limited to the use of any copies of copyrighted images that they may create
from the dataset."

The per-image Flickr licence is a field in the annotation file. In the sampled 500 images it
ranges over CC BY-NC-SA, BY-NC-ND, BY, BY-NC, BY-SA and BY-ND 2.0 (counts: 151, 143, 78, 61, 40,
27). Each gallery row carries its licence and Flickr URL in `licence_note`.

Hashes of what was downloaded:

| File | Bytes | sha256 |
|---|---|---|
| `annotations_trainval2017.zip` | 252,907,541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `captions_val2017.json` (from the zip) | 3,872,473 | `afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9` |

### 2. TextCaps 0.1 (reading slice)

| Item | URL | Status | Size | Terms |
|---|---|---|---|---|
| Validation annotations | `https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json` | 200 | 25,518,287 B | CC BY 4.0 |
| Dataset page | `https://textvqa.org/textcaps/dataset/` | 200 | | read live |
| Images, fetched by Open Images id | `https://open-images-dataset.s3.amazonaws.com/train/<image_id>.jpg` (CVDF mirror) | 200 (sampled) | 100-170 kB each | Open Images: listed CC BY 2.0 |
| Open Images licence statement | `https://storage.googleapis.com/openimages/web/factsfigures_v4.html` | 200 | | read live |

Terms as published on the TextCaps page: "Data is available under CC BY 4.0 license." The
validation set is 15,830 captions over 3,166 images (5 per image; each JSON row repeats the same
five `reference_strs`, which this repo de-duplicates by image). "Images for training and
validation set are from OpenImages train set." Open Images states: "The annotations are licensed
by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license." and
adds that it makes no representation about each image's licence status. The JSON also carries
`flickr_original_url` and `flickr_300k_url`, both of which resolved (200) for the one id tried.
The CVDF mirror resolved (200) for the 3 random sampled gallery ids in the dry run, and
`validation/<id>.jpg` returns 404 for these ids, which is expected: they are in the `train/` split.

The page also warns that some Open Images files are rotated and tells users to check the Rotation
field of the Open Images id files. This script reads EXIF orientation from each downloaded file
and the count is in "Download results" below; the embedding step must apply it.

| File | Bytes | sha256 |
|---|---|---|
| `TextCaps_0.1_val.json` | 25,518,287 | `1909abda82adefc7e998f5bd16b4bc03d0326507676b4de7c57499ad1b867f22` |

### 3. Model weights (later steps)

CLIP ViT-B/32, OpenAI weights, MIT licence. Not downloaded by this script.

## Gallery construction

- `scripts/fetch_gallery.py select`: seeded sample (seed 47; ids sorted, then shuffled with
  `random.Random("47-<source>")`). COCO: 5,000 images in val2017 all have at least one caption;
  TextCaps: 3,166 images. First 500 of each shuffled list.
- Image ids are prefixed so the two sources cannot collide: `coco_<12-digit id>` and
  `tc_<Open Images id>`. Query ids: `coco-<annotation id>` and `tc-<Open Images id>-<0..4>`.
- Captions: whitespace collapsed and trimmed, nothing else changed (typos, casing and trailing
  full stops are kept).
- `scripts/fetch_gallery.py download`: 8 parallel connections, 4 tries with backoff, resumable.
  A permanently failing or duplicate-content image is replaced by the next id in the same seeded
  order, with its queries swapped consistently; replacements are logged in
  `data/local/replacements.jsonl`.

## What is committed, and what is not

| Committed | Not committed (git-ignored `data/local/`) |
|---|---|
| image ids, source, file name, URL, sha256, width, height, licence note (`data/gallery.jsonl`) | any image file |
| human captions as queries (`data/queries.jsonl`) | the annotation zips and JSON files |
| the blind tagging sheet (captions, opaque ids) | copies of images used for tagging |
| later: OCR text and CLIP embeddings derived from the images | model weights |

## Attribution required

- COCO annotations (captions): COCO Consortium, CC BY 4.0. Cite Lin et al., "Microsoft COCO:
  Common Objects in Context", ECCV 2014, and link the licence.
- TextCaps annotations (captions): Meta (Facebook AI Research) authors, CC BY 4.0. Cite Sidorov
  et al., "TextCaps: a Dataset for Image Captioning with Reading Comprehension", ECCV 2020.
- Images: owned by their Flickr authors; the Flickr URL and licence of each image is in
  `licence_note` in `data/gallery.jsonl`. Images are fetched at run time and never redistributed.
- Open Images: Kuznetsova et al., "The Open Images Dataset V4", IJCV 2020 (annotations CC BY 4.0).

## Residual risk

COCO images are individually licensed, and 382 of the 500 sampled ones (76%) carry a non-commercial
(NC) or no-derivatives (ND) Flickr licence. This is a non-commercial research and portfolio
project, images are not redistributed, and only text and numeric vectors derived from them are
committed. A vector that cannot reproduce the image is a different thing from a copy of it, but
the licences do not address this case explicitly. If the committed embeddings are ever judged a
problem, the fallback is to commit TextCaps and COCO-BY-only embeddings, or none, and have
users re-run the local embedding step.

## Download results

Run 2026-10-25 with `scripts/fetch_gallery.py download` (8 workers):

| | COCO | TextCaps | Total |
|---|---|---|---|
| Images | 500 | 500 | 1,000 |
| Queries (captions) | 2,502 | 2,500 | 5,002 |
| Captions per image | 5.00 | 5.00 | |
| Bytes on disk | 80,503,307 | 142,821,326 | 223,324,633 |
| Longest side (max) | 640 | 1,024 | |

- Failures and replacements: 0. All 1,000 sha256 values are distinct.
- Pixel sizes read from the downloaded files equal the sizes in the annotations for all 1,000.
- 6 grayscale files; 0 files with an EXIF rotation flag.
- Wall time about 5 minutes; fetches of 1,000 images use 1,000 requests plus the two annotation files.
