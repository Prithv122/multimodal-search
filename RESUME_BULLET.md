# Resume Bullets — multimodal-search

Form: **action → technical specifics → measured outcome.** All numbers are from
`results/test/eval.json` (800 held-out images, 4,002 queries, scored once) unless marked
exploratory.

---

## Bullets

- **AI engineer angle.** Built a text-to-image search system fusing CLIP ViT-B/32 embeddings with
  BM25 over PaddleOCR text on a 1,000-image gallery (COCO + TextCaps, 5,002 caption queries);
  two-stage design lets CI replay committed vectors with numpy and scipy only. A dev-tuned
  weighted fusion gained +0.066 nDCG@10 on the TextCaps slice (95% CI +0.054 to +0.079), below
  the pre-registered +0.10 bar, at a cost of 0.008 on the COCO slice, while plain RRF lost 0.440
  there.
- **Data science angle.** Pre-registered a hypothesis test before embedding any data (arms, split,
  pass thresholds), then scored the held-out split once with a paired cluster bootstrap over
  images (20,000 resamples). The headline hypothesis failed (+0.066 against a +0.10 bar) and was
  reported as a failure; CLIP alone scored 0.748 on the TextCaps slice, level with 0.745 on COCO.
- **Evaluation-rigour angle.** Caught a slice-correlated tie-breaking bias (image-id and file
  order) in code review on synthetic fixtures before any real number existed, fixed it with a
  per-query hash, and logged it as a dated amendment; tested the pipeline against planted, null,
  filler and shuffled fixtures, and added a replay command that recomputes the committed results
  in memory.

## Which roles this supports

- [x] Data Scientist / ML
- [x] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [ ] Data Analyst / Python Developer

## Keywords this project earns

Only what the repo uses and can be questioned on: CLIP (ViT-B/32, `transformers`), PaddleOCR
(PP-OCRv5), BM25, reciprocal rank fusion, score fusion with min-max normalisation, nDCG@10,
recall@k, MRR@100, cluster bootstrap, paired differences, pre-registration, train/dev/test
split by group, exact cosine search, synthetic fixtures, `uv`, `pytest`, `ruff`, Gradio.

---

### Bad vs good

Bad: "Built an AI image search engine using CLIP and OCR."
Good: the first bullet above. It names the data size, the arms, the interval and the cost.

Do not write "improved search by 8.8%". The honest figure is +0.066 nDCG@10 on one slice of one
gallery, below the pre-registered bar, with a small cost on the other slice.
