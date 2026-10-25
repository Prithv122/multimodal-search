# multimodal-search — H5

**Tier:** 3 · **Category:** H — Computer vision · **Wave:** 4

The project's own engineering guide. Keep it short and current.

## What this is

Text-to-image search that can read. CLIP matches what an image shows and reads text inside it
poorly; an OCR text index has the opposite profile. This project measures that gap on a
1,000-image gallery (COCO = visual slice, TextCaps = reading slice) and tests whether fusing the
two closes it without costing the visual slice. The plan and pass rules are in `EVAL_PLAN.md`.

## Stack

- Python 3.12, `uv`. Runtime: numpy + scipy only (dense index, BM25, fusion, metrics, bootstrap).
- `embed` extra (local only): CLIP ViT-B/32 (OpenAI weights) and PaddleOCR 3.x.
- `demo` extra: Gradio app over the committed arrays.
- Index, BM25, fusion and metrics are vendored from `production-rag`; provenance is in each
  file's header under `src/mmsearch/_vendor/`.

## Acceptance criteria

- [x] The pre-registered H1 and H2 are scored once on the test split and reported as written (H1 fail, H2 pass)
- [x] Every README number comes from `results/` files produced by the committed code and arrays
- [x] The synthetic planted fixture is detected and the null fixture is not (tested in CI)
- [x] The package never imports torch; CI asserts it
- [ ] Runs from a clean clone using only the documented steps
- [ ] Tests, lint and CI green

## Conventions

- `uv` for dependencies; `ruff check` + `ruff format`; `pytest`.
- No secrets in the repo; this project reads no environment variables.
- **Images are never committed.** Only ids, captions, URLs, hashes, OCR text and embeddings.
- The test split is scored once. Anything changed after a test number is seen is an amendment
  in `EVAL_PLAN.md`, dated, with the reason.

## Project-specific notes

- Two stages so CI never needs a model: `scripts/fetch_gallery.py` + `scripts/embed_and_ocr.py`
  run locally and write `data/`; everything in `src/mmsearch` replays those arrays.
- Downloads live in the git-ignored `data/local/`.
- The slice is the source dataset, not the query type. The owner's hand tags in `data/tags.csv`
  are the robustness check for that confound. They are not applied yet: the 100-query blind
  sheet is in `data/tagging/`, and `mmsearch tagged` regroups the saved per-query rows without
  re-ranking.
- Open items: hand tags, an OCR word-recall hand check, a hosted demo and the public repository.
  `scripts/explore.py` writes the labelled-exploratory table in `results/test/exploratory.json`.
