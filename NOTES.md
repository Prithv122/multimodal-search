# Build Notes — multimodal-search

Working notes: what broke, what was tried, why X over Y. Written to be useful in an interview
six months from now. Numbers are copied from `results/` files.

---

## Log

### Pre-registration first
- **Did:** wrote `EVAL_PLAN.md` (arms, split, metrics, H1 and H2, pass thresholds, what counts as
  an allowed outcome) and committed it before any image was embedded or any OCR was run.
- **Why:** the project's whole point is a measurement. If the thresholds are chosen after the
  numbers, a +0.066 gain could have been called a win. It was +0.066 against a bar of +0.10, so
  H1 failed, and the plan said that was a reportable result.

### Tie-breaking was biased by the slice (Amendment A1)
- **Broke:** an independent code review, run on synthetic fixtures only, found that tied scores
  were ordered by something that tracks the slice under test. The vendored fusion breaks ties by
  image id, and every `coco_...` id sorts before every `tc_...` id. The BM25 top-k selection left
  tied documents in near-reverse file order, and the gallery file lists COCO first.
- **Why it matters:** exact ties are common. In RRF an image found only by CLIP at rank r ties
  with an image found only by OCR at rank r. Left alone, id spelling would push the visual and
  the reading slice in opposite directions with no model involved.
- **Fixed by:** sort every base list and every fused list by (score descending, per-query hash
  of query id and image id) before truncating. Fusion is asked for the whole candidate union,
  then the result is re-sorted and cut. Tests reorder the gallery file, swap the id spelling of
  the two sources, and check that constructed cross-source ties are won about half the time by
  each side.
- **Process:** recorded as Amendment A1 in the plan before any real dev or test number existed,
  so it is a fix, not a post-hoc change. MRR was relabelled MRR@100 in the same pass (a gold
  image outside an arm's top 100 scores 0).
- **Learned:** if the thing you break ties with correlates with the thing you measure, you
  measure the tie-break. Review the pipeline on data where you know the answer, before you
  look at real numbers.

### The CLI scored test by default
- **Broke:** `eval` defaulted to the test split. One careless run would have used up the single
  look.
- **Fixed by:** default is now dev. `--split test` writes `results/test/eval.json` with an
  exclusive create and is refused once that file exists. A custom `--out` and `--split all` are
  refused on real data because either would score test with no record.

### Synthetic fixtures: planted, null, filler, shuffled
- **Tried:** a planted effect (OCR-only words in reading-slice captions) and a null (OCR text
  drawn independently of captions). Both passed.
- **Then added `filler`:** common words ("a", "the", "of") in every caption and in OCR text. The
  planted gain survived on the reading slice, but fusion cost the visual slice, because a weak
  OCR hit from filler words is min-max scaled up to 1.0. That predicted that un-weighted fusion
  would hurt the visual slice on real data. It did: equal-weight score fusion −0.227, RRF
  −0.440 nDCG@10 against CLIP.
- **And `shuffled`:** reading-slice OCR text moved to other images, so BM25 is confidently wrong.
  The dev-chosen weight retreats towards CLIP and a fusion pinned at 0.5 fails H2. That is the
  behaviour wanted from a weight chosen on dev.
- **Learned:** a fixture that is too clean only proves the code runs. The filler fixture was the
  one that predicted something.

### OCR: mobile models, thread count, a shared CPU
- **Chose:** PP-OCRv5 mobile detector and recogniser, no preprocessing. The script also has a
  `server` model flag and a CLAHE flag; neither was run for the reported numbers.
- **Broke:** OCR throughput on a loaded, shared machine was poor with high thread counts
  or oneDNN. Measured on this machine (12 threads, shared with other jobs): 2 to 4 threads ran about 1.3 to
  1.7 s per image on the mobile models, while 6 to 8 threads or oneDNN were 2 to 4 times slower.
  The script now bounds threads and turns oneDNN off. CLIP weights are loaded lazily.
- **Result:** the full OCR pass took about 2.5 hours of wall time. CLIP took about 8 minutes.
- **Not done:** no comparison against the server models, so I do not know whether a better OCR
  model would change H1. It is the first thing to try if this is reopened.

### Saving per-query rows so tags do not need a second look at test
- **Problem:** the confound (source is not query type) needs a hand-tagged subset. Tags arrive
  after the test split was scored. Re-running `eval --split test` is forbidden.
- **Fixed by:** every scored split now saves per-query metric rows (`results/*/per_query.csv`).
  `mmsearch tagged` regroups those rows by tag and recomputes the paired difference
  `best fusion − clip`. It ranks nothing and chooses nothing. The same rows fed
  `scripts/explore.py`.

### The main surprise: CLIP already reads
- **Expected:** a clear gap, CLIP weak on the reading slice, OCR filling it.
- **Got:** CLIP nDCG@10 0.748 on TextCaps against 0.745 on COCO. OCR-BM25 alone reached 0.419 on
  TextCaps, and found a hit that CLIP missed on only 70 of 2,000 reading queries
  (`results/test/exploratory.json`).
- **Possible reasons, none tested:** TextCaps captions often name the scene as well as the
  text; CLIP ViT-B/32 can read some large, clear words; BM25 on noisy mobile OCR is a weak
  signal. The hand tags are the way to separate the first from the rest.

### Report discipline
- H1 is reported as a fail, and the README says the interval excludes 0 but sits entirely below
  the 0.10 bar.
- H2 is reported as a pass, with the note that the interval is wholly below zero, so the cost on
  the visual slice is small but not nothing.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| Score test several times while developing | The plan's one non-negotiable rule: test is scored once |
| Pick "best fusion" on test | Best-of-three on test inflates H1; picked on dev instead |
| A different weight per slice | A real query has no slice label |
| Bootstrap over queries | The captions of an image are not independent; images are resampled instead |
| Breaking ties by image id or file order | Tracks the slice (Amendment A1) |
| `open_clip_torch` | Same OpenAI weights; `transformers` reports token lengths and avoids `timm`/`torchvision` |
| ANN index | 1,000 vectors; exact search is faster to reason about and has no approximation error |
| Tune RRF's k | Pre-registered at 60; the standard setting is part of the finding |
| OCR preprocessing (CLAHE) | Fixed to none in the plan; natural scenes, not scans |

## Open questions

- [ ] Owner hand tags (100 blind queries; sheet in `data/tagging/`). Then `uv run mmsearch tagged`.
      Does the fusion gain grow on queries tagged `text`?
- [ ] OCR word recall on a hand-checked sample of reading-slice images.
- [ ] Would the PP-OCRv5 server models, or a confidence floor (`--min-confidence`), change H1?
- [ ] Hosted demo and a public repo are not done.
