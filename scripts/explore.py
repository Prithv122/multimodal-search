"""EXPLORATORY, post hoc, not pre-registered: where fusion helps and where it hurts.

Reads three committed files and nothing else:

* ``results/test/per_query.csv``  per-query metrics saved when the test split was scored once;
* ``data/queries.jsonl``          the captions;
* ``data/ocr/ocr.jsonl``          the OCR text of every gallery image.

It ranks nothing and scores nothing new. Every "hit" below is ``recall@10 == 1`` read from the
saved per-query rows, so the test split is not looked at a second time in the sense of the plan:
the same numbers are only regrouped. The one question that the saved rows cannot answer, "does
the OCR arm return anything at all for a caption?", is answered from the tokenizer alone: a
caption can only retrieve an image through BM25 if it shares at least one token with that
image's OCR text (the arm applies no stopword list). That is a set intersection, not a ranking.

Run from the repository root::

    uv run python scripts/explore.py

Writes ``results/test/exploratory.json``. Because nothing was pre-registered for these
questions, treat every number as descriptive. They suggest an explanation; they do not test it.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from mmsearch._vendor.bm25 import tokenize

ROOT = Path(__file__).resolve().parents[1]
FUSION = "fuse_score_weighted"
OTHER_FUSIONS = ("fuse_rrf", "fuse_score")
SLICES = ("coco", "textcaps")


def _read_per_query(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _hit(row: dict, arm: str) -> bool:
    return float(row[f"{arm}:recall@10"]) >= 1.0


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _overlap_table(rows: list[dict], a: str, b: str) -> dict:
    """Four-way split of queries by whether arm ``a`` and arm ``b`` hit in the top 10."""
    both = sum(1 for r in rows if _hit(r, a) and _hit(r, b))
    only_a = sum(1 for r in rows if _hit(r, a) and not _hit(r, b))
    only_b = sum(1 for r in rows if not _hit(r, a) and _hit(r, b))
    neither = len(rows) - both - only_a - only_b
    n = len(rows)
    return {
        "n_queries": n,
        "both_hit": both,
        f"only_{a}_hits": only_a,
        f"only_{b}_hits": only_b,
        "neither_hits": neither,
        "share_both": _rate(both, n),
        f"share_only_{a}": _rate(only_a, n),
        f"share_only_{b}": _rate(only_b, n),
        "share_neither": _rate(neither, n),
    }


def _rescue_break(rows: list[dict], fusion: str) -> dict:
    """Against ``clip``: queries the fusion rescues (clip misses, fusion hits) or breaks."""
    clip_miss = [r for r in rows if not _hit(r, "clip")]
    clip_hit = [r for r in rows if _hit(r, "clip")]
    rescued = sum(1 for r in clip_miss if _hit(r, fusion))
    broken = sum(1 for r in clip_hit if not _hit(r, fusion))
    return {
        "n_queries": len(rows),
        "clip_misses_top10": len(clip_miss),
        "clip_hits_top10": len(clip_hit),
        "rescued": rescued,
        "broken": broken,
        "net": rescued - broken,
        "rescued_share_of_clip_misses": _rate(rescued, len(clip_miss)),
        "broken_share_of_clip_hits": _rate(broken, len(clip_hit)),
    }


def main() -> None:
    rows = [r for r in _read_per_query(ROOT / "results/test/per_query.csv") if r["split"] == "test"]
    queries = {
        q["qid"]: q
        for q in map(json.loads, (ROOT / "data/queries.jsonl").read_text("utf-8").splitlines())
    }
    ocr_docs: dict[str, str] = {}
    image_source: dict[str, str] = {}
    for line in (ROOT / "data/ocr/ocr.jsonl").read_text("utf-8").splitlines():
        rec = json.loads(line)
        text = " ".join(" ".join(x["text"].split()) for x in rec["lines"] if x["text"].strip())
        ocr_docs[rec["image_id"]] = text
    for line in (ROOT / "data/gallery.jsonl").read_text("utf-8").splitlines():
        rec = json.loads(line)
        image_source[rec["image_id"]] = rec["source"]

    by_slice = {s: [r for r in rows if r["source"] == s] for s in SLICES}
    out: dict = {
        "label": "EXPLORATORY: post hoc, not pre-registered, descriptive only",
        "inputs": [
            "results/test/per_query.csv",
            "data/queries.jsonl",
            "data/ocr/ocr.jsonl",
            "data/gallery.jsonl",
        ],
        "hit_definition": "recall@10 == 1 for the query's own image, read from per_query.csv",
        "fusion_arm": FUSION,
        "n_test_queries": len(rows),
    }

    # 1. Complementarity of clip and ocr_bm25, per slice.
    out["clip_vs_ocr_bm25"] = {s: _overlap_table(by_slice[s], "clip", "ocr_bm25") for s in SLICES}

    # 2. Does the fusion rescue a clip miss or break a clip hit, per slice and per fusion arm.
    out["rescue_break_vs_clip"] = {
        arm: {s: _rescue_break(by_slice[s], arm) for s in SLICES}
        for arm in (FUSION, *OTHER_FUSIONS)
    }

    # 3. OCR coverage of the gallery, and of the images behind the test queries.
    coverage: dict = {}
    for s in SLICES:
        ids = [i for i, src in image_source.items() if src == s]
        with_text = sum(1 for i in ids if ocr_docs.get(i, "").strip())
        coverage[s] = {
            "images": len(ids),
            "images_with_any_ocr_text": with_text,
            "share": _rate(with_text, len(ids)),
        }
    out["ocr_coverage_gallery"] = coverage

    # 4. Reading slice: how often the gold image has OCR text at all, and clip's hit rate then.
    tc = by_slice["textcaps"]
    has_text = [r for r in tc if ocr_docs.get(r["image_id"], "").strip()]
    no_text = [r for r in tc if not ocr_docs.get(r["image_id"], "").strip()]
    out["reading_slice_by_gold_image_ocr"] = {
        "queries_whose_image_has_ocr_text": len(has_text),
        "queries_whose_image_has_no_ocr_text": len(no_text),
        "clip_top10_hit_rate_when_text": _rate(
            sum(_hit(r, "clip") for r in has_text), len(has_text)
        ),
        "clip_top10_hit_rate_when_no_text": _rate(
            sum(_hit(r, "clip") for r in no_text), len(no_text)
        ),
        "ocr_top10_hit_rate_when_text": _rate(
            sum(_hit(r, "ocr_bm25") for r in has_text), len(has_text)
        ),
        "fusion_ndcg10_minus_clip_when_text": round(
            sum(float(r[f"{FUSION}:ndcg@10"]) - float(r["clip:ndcg@10"]) for r in has_text)
            / len(has_text),
            3,
        ),
        "fusion_ndcg10_minus_clip_when_no_text": round(
            sum(float(r[f"{FUSION}:ndcg@10"]) - float(r["clip:ndcg@10"]) for r in no_text)
            / len(no_text),
            3,
        )
        if no_text
        else None,
    }

    # 5. Why plain RRF hurts the visual slice: OCR returns a list for almost every caption.
    #    A caption retrieves an image through BM25 only if it shares a token with its OCR text.
    #    Tokens come from the vendored tokenizer; there is no stopword list in the arm.
    doc_tokens = {i: set(tokenize(ocr_document_text)) for i, ocr_document_text in ocr_docs.items()}
    token_df: Counter[str] = Counter()
    for toks in doc_tokens.values():
        token_df.update(toks)
    visual_stats = {}
    for s in SLICES:
        n_any = 0
        n_gold = 0
        n_images_matched = []
        shared_counter: Counter[str] = Counter()
        for r in by_slice[s]:
            q_tokens = set(tokenize(queries[r["qid"]]["text"]))
            matched = [i for i, toks in doc_tokens.items() if q_tokens & toks]
            if matched:
                n_any += 1
            n_images_matched.append(len(matched))
            if q_tokens & doc_tokens.get(r["image_id"], set()):
                n_gold += 1
            for t in q_tokens:
                if token_df.get(t):
                    shared_counter[t] += 1
        n_images_matched.sort()
        mid = len(n_images_matched) // 2
        visual_stats[s] = {
            "n_queries": len(by_slice[s]),
            "queries_sharing_a_token_with_any_gallery_ocr_text": n_any,
            "share_sharing_a_token_with_any_ocr_text": _rate(n_any, len(by_slice[s])),
            "queries_sharing_a_token_with_their_own_image_ocr_text": n_gold,
            "share_sharing_a_token_with_own_image_ocr_text": _rate(n_gold, len(by_slice[s])),
            "median_images_matched_per_query": n_images_matched[mid],
            "top_query_tokens_found_in_ocr_text": [
                {"token": t, "queries": c, "gallery_images_containing_it": token_df[t]}
                for t, c in shared_counter.most_common(8)
            ],
        }
    out["ocr_returns_something"] = visual_stats

    # 6. The same on per_query terms: gold in the OCR list at all (mrr@100 > 0) per slice.
    out["ocr_gold_in_top100_share"] = {
        s: _rate(sum(float(r["ocr_bm25:mrr@100"]) > 0 for r in by_slice[s]), len(by_slice[s]))
        for s in SLICES
    }

    path = ROOT / "results/test/exploratory.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
