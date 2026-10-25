# Local demo

A small Gradio app over the gallery: type a query, pick one of the five arms, and see the top
images with their rank, score, source and image id, plus the OCR text read from each hit.

This is a local demo. There is no hosted version yet.

## Run

From the repository root:

```bash
uv run --extra embed --extra demo python demo/app.py
```

Then open <http://127.0.0.1:7860> (`--port N` changes it). The `embed` extra supplies the CLIP
text encoder (torch and transformers); the `demo` extra supplies Gradio.

The first free-text query needs the CLIP weights (`openai/clip-vit-base-patch32`, downloaded
once and cached by Hugging Face). The app loads them in the background at start-up.

## What it shows

- **Query and arm.** `clip`, `ocr_bm25`, `fuse_rrf`, `fuse_score`, `fuse_score_weighted`, with a
  top-k slider (default 8). The weighted arm uses the CLIP weight recorded in a file under
  `results/dev/` when one exists; otherwise it uses 0.5 and the page says so.
- **Results.** A gallery of the top-k images, each labelled `#rank | score | source | image id`,
  and a table with the OCR text of each hit.
- **Example captions.** A dropdown of captions from `data/queries.jsonl` (a few from COCO, a few
  from TextCaps). A query that is exactly a stored caption uses its stored embedding, so the
  examples work without the `embed` extra and without any weight download.
- **Scores are per arm.** CLIP shows cosine similarity, BM25 its raw score, RRF and the score
  fusions their fused values. They are comparable within one ranking, not across arms.

## Images

No images are stored in the repository. Your browser fetches each one from its original host
(COCO from `images.cocodataset.org`, TextCaps from the Open Images S3 bucket), so the gallery needs an internet
connection and an image can fail to load if its host removes it.

## When OCR is missing

If `data/ocr/ocr.jsonl` is absent, or does not cover every gallery image (for example, an OCR run that has
not finished), the app shows a notice and runs CLIP-only: the OCR and fusion arms return the CLIP
ranking and say so. A partial file is not used, because an image that has not been read yet
would look the same as an image with no text.

## Search from the command line with the same encoder

`demo/encoder.py` is a CLIP text encoder with `encode_query(text)`, in the same space as
`data/embeddings/clip_img.npy` (same checkpoint, tokenizer, 77-token truncation and L2
normalisation as the script that wrote the arrays). It plugs into the search command:

```bash
uv run --extra embed python -m mmsearch.cli search "a dog catching a frisbee" --arm clip --encoder demo.encoder:factory
```

Use `python -m`, not the `mmsearch` script: `-m` puts the working directory on `sys.path`, which
is how `demo.encoder` is found. The `src/mmsearch` package itself never imports torch or gradio.
