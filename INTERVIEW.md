# Interview Prep — multimodal-search

**Five questions, five answers.** Numbers are from `results/test/eval.json` unless stated.
Exploratory numbers are from `results/test/exploratory.json` and are marked as such.

---

### Q1. Your main hypothesis failed. Is that a bad result?

**A.** The hypothesis (H1) was that the best fusion would beat CLIP by at least 0.10 nDCG@10 on
the reading slice, with an interval that excludes 0. The gain was +0.066, interval
[+0.054, +0.079]. So the gain is real (the interval excludes 0) but smaller than the bar, and
the whole interval is under 0.10. H1 fails. The second check, H2, passed: on the visual slice
the fusion was −0.008 [−0.013, −0.003] against a tolerance of 0.02.

It is not a bad result, for two reasons. First, I wrote the plan and the thresholds before any
image was embedded, and the plan listed this outcome as allowed. A pre-registration that can
only pass is not a test. Second, it taught something: CLIP ViT-B/32 already scored 0.748 on the
reading slice, level with 0.745 on the visual slice, so there was less for OCR to add than the
premise assumed. Exploratory counts agree: OCR hit the top 10 where CLIP missed on only 70 of
2,000 reading queries.

What I would not say: that fusion is worthless, or that OCR does not matter. The reading slice
is a dataset, not a set of queries that need reading. I would say that on caption queries over
1,000 images, a tuned fusion gives a small, real gain on text-heavy images and a small, real cost
on the rest.

---

### Q2. Why did plain RRF lose to CLIP alone, when RRF is the standard hybrid-search method?

**A.** RRF (reciprocal rank fusion, k = 60) uses only ranks. A document at OCR rank 1 contributes
as much as a document at CLIP rank 1, whatever the scores were. That is right when both
retrievers are decent on every query. Here the OCR arm is nearly useless on the visual slice
(nDCG@10 0.019), yet it returns a ranked list for every caption: in the exploratory analysis
every COCO test caption shared at least one token with the OCR text of some gallery image
(median 102 images), mostly function words such as "a" and "the", while only 0.043 shared a
token with their own image. RRF gave those spurious OCR ranks full weight. On the visual slice it
scored −0.440 against CLIP, and in the exploratory table it broke 386 of 1,858 queries CLIP had
right in the top 10.

Score fusion is better only if the weights are. With equal weights it still cost 0.227 on the
visual slice. A dev-tuned weight of 0.70 on CLIP made it nearly harmless there (−0.008). I
should be careful about the mechanism: the filler-word explanation fits the data and was
predicted by a synthetic fixture, but I did not run an experiment that removes stopwords from
the OCR arm. That would be the direct test.

The same thing happened in my earlier production-rag project, where plain RRF lost to BM25.
The lesson is the same: fusion needs the arms to be roughly comparable in quality, or a weight
that says they are not.

---

### Q3. Why a cluster bootstrap over images and not an ordinary bootstrap over queries?

**A.** Each image has five captions, and they are not independent. If the image is hard for CLIP,
all five captions tend to be hard together. A bootstrap over queries treats 4,002 test queries as
4,002 independent draws and reports intervals that are too narrow. Resampling images with
replacement, each image carrying all its captions, keeps that dependence. The effective sample
is 800 images, not 4,002 queries.

I also paired the arms: both arms are scored on the same resample, so the interval for
"fusion minus CLIP" is the interval of the difference. Comparing two overlapping intervals of
separate means throws away the pairing and is much less sensitive. This is why a difference of
0.008 on the visual slice can have an interval of [−0.013, −0.003] while the two arm intervals
overlap heavily.

Limits: the weight was tuned on 200 dev images and held fixed on test, so the test intervals do
not include the uncertainty of choosing it. I used 20,000 resamples and a fixed seed so the
interval is reproducible; `mmsearch replay` recomputes it and checks it matches.

---

### Q4. Your slices are datasets, not query types. How do you deal with that?

**A.** It is a confound and I said so in the plan. "Reading slice" means TextCaps, whose images
contain text. It does not mean the caption needs the text. A TextCaps caption like "a red bus on a
city street" can be answered by looking. If so, a null result on the reading slice says little
about queries that do need reading.

What I did about it, in order: (1) reported results by source because that is what was
pre-registered and measurable, and labelled it as the dataset throughout; (2) built a blind
tagging sheet: 100 queries, 50 per source, source hidden, tagged visual / text / both by the
owner; (3) designed the pipeline to save per-query metric rows for each scored split, so that
`mmsearch tagged` can regroup them by tag and recompute `best fusion − clip` without ranking or
scoring anything again.

That last part matters. The test split is scored once. If I had to re-run it to apply tags, I
would break my own rule. The tags are not done yet and the README says so; the project does not
claim anything about queries that truly need reading. The exploratory split by whether the gold
image has OCR text at all (1,985 of 2,000 reading queries do) shows the confound clearly: almost
every TextCaps image has text, but CLIP still hits the top 10 on 0.889 of those queries.

---

### Q5. How would you ship this in production, and what breaks first?

**A.** Architecture: ingest is offline. For each new image, run CLIP once and OCR once, store the
vector and the text keyed by content hash. At query time, encode the query with CLIP, do a
nearest-neighbour search, run BM25 over the OCR text, and fuse with a weight. The design here is
already split that way: the heavy stage writes arrays, the serving stage reads them.

What breaks first, in order:

1. **OCR cost at ingest.** It took about 2.5 hours for 1,000 images on a shared CPU, against about
   8 minutes for CLIP. I would use GPU batching, skip images where a cheap text detector finds
   nothing (only 192 of 500 COCO images had any text), and cache by content hash.
2. **Exact search.** Fine at 1,000 vectors. At millions I would move to an approximate index and
   measure recall loss against exact search, since that approximation error is not in these results.
3. **Fusion weight drift.** One global weight of 0.70 was tuned on 200 images. In production, the
   mix of queries decides the best weight, so I would re-tune it on a labelled sample from real
   traffic and monitor it.
4. **Query shape.** My queries are captions. Real queries are shorter and vaguer, and CLIP's edge
   partly comes from matching caption style. I would build an eval set from real logs before
   trusting any of these numbers.

Honest scope: given the measured result, I would ship CLIP alone first and add the OCR arm only
after it showed a gain on a query set that really needs reading. The gain I measured is +0.066
nDCG@10 on caption queries, at the cost of an OCR pipeline.

---

## 30-second pitch

CLIP is said to read text in images poorly, so I tested whether adding an OCR text index fixes it.
I pre-registered the plan, built a 1,000-image gallery from COCO and TextCaps, and scored a held-out
test split once. A tuned fusion gave +0.066 nDCG@10 on the reading slice, below my +0.10 bar, and
cost 0.008 on the visual slice, while plain rank fusion hurt badly. The surprise was that CLIP
alone already scored 0.748 on the reading slice. I reported it as a failed hypothesis with
intervals and an honest list of what the data cannot show.
