# Evaluation plan — pre-registered

Written and committed **before** any image was embedded, any OCR was run, or any metric was
computed on real data. The git history shows this file predates `data/embeddings/` and
`data/ocr/`. Anything decided after the first look at real numbers goes in an amendment section
at the bottom, dated, with the reason. This section is never edited after the run.

## Question

On a 1,000-image gallery, how do CLIP-only, OCR-BM25-only and their fusion compare in nDCG@10 for
queries about visual content versus queries that need reading text in the image, and does the
best fusion beat the best single arm on each slice?

## Data

- Gallery: 500 COCO val2017 images (the **visual** slice) + 500 TextCaps validation images (the
  **reading** slice), sampled with seed 47. Images are not committed; ids, captions, URLs and
  hashes are (see `DATA.md`).
- Queries: every human caption of every gallery image. Each query has exactly one relevant item,
  its own image. A caption that happens to describe another image as well still counts only its
  own image as relevant; this makes every score a slight underestimate, equally for all arms.

## Arms (all fixed before the run)

| Arm | What it ranks by |
|---|---|
| `clip` | Cosine similarity of the CLIP ViT-B/32 (OpenAI weights) caption embedding to each image embedding |
| `ocr_bm25` | BM25 of the caption against the image's OCR text (PaddleOCR 3.x, no preprocessing) |
| `fuse_rrf` | Reciprocal rank fusion of the two rankings |
| `fuse_score` | Equal-weight sum of per-query normalised scores |
| `fuse_score_weighted` | Weighted sum; the weight is chosen on the dev split only |

Each arm returns its top 100 so fusion has depth. RRF uses the vendored constant (k = 60).

## Split

Images, not queries, are split: a seeded hash of the image id puts ~20% of images (stratified by
source) in **dev** and ~80% in **test**. All captions of an image go with it. The weight for
`fuse_score_weighted` and the choice of "best fusion" are made on dev. **Test is scored once.**

## Metrics and intervals

recall@1, recall@5, recall@10, nDCG@10, MRR, per slice. 95% intervals come from a cluster
bootstrap over images (images resampled with replacement, carrying all their captions; 20,000
resamples, fixed seed). Differences between arms are paired: both arms are scored on the same
resample.

## Hypotheses and pass rules

**Best fusion** = the fusion arm with the highest mean nDCG@10 on the **dev** split (both slices
pooled). It is fixed before test is scored, so the test comparison is not a best-of-three pick.

- **H1 (reading slice).** On the test split, reading slice: best fusion − `clip` ≥ **0.10**
  nDCG@10, and the 95% interval of that paired difference excludes 0. Both conditions are needed.
- **H2 (visual slice).** On the test split, visual slice: |best fusion − `clip`| ≤ **0.02**
  nDCG@10 (point estimate). The interval is reported alongside.

## Secondary, reported but not pass/fail

- The full arm × slice × metric table, and whether the best fusion beats the best *single* arm on
  each slice (the brief's second question).
- **Tagged subset.** The slice is the source dataset, not the query type; that is a confound. The
  owner hand-tags 100 queries (50 per source, source hidden) as visual / text / both, and the same
  table is reported on that subset. If the tags are not done, the report says so.
- **OCR quality.** OCR word recall on a hand-checked sample of reading-slice images, if the owner
  checks one; otherwise the share of images with any OCR text, per slice.

## Allowed outcomes

All of these are reportable as results, not failures of the project:

- Fusion helps on reading and costs nothing on visual (H1 and H2 pass).
- Fusion helps on reading but hurts visual (H1 passes, H2 fails).
- Fusion does not beat the best single arm (as plain RRF lost to BM25 in production-rag).
- OCR is too noisy to help.

## Not allowed

Changing the model, OCR engine or preprocessing, k, the weight grid, the split seed or the pass
thresholds after seeing a test number; scoring test more than once; tuning anything on test.

## Amendments

_None yet._
