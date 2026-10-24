"""Tied scores must not be broken by anything that tracks the slice under test."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mmsearch.arms import FUSION_ARMS, Retriever, fuse_rankings, rank_queries
from mmsearch.evaluate import NDCG, EvalConfig, metrics_matrix, run_study
from mmsearch.gallery import Gallery, GalleryItem, OcrLine, OcrRecord
from mmsearch.index import MultimodalIndex, TieBreaker


def _reordered(g: Gallery) -> Gallery:
    """The same data with gallery items, queries and embedding rows in reverse file order."""
    emb = replace(
        g.embeddings,
        image_vectors=g.embeddings.image_vectors[::-1].copy(),
        query_vectors=g.embeddings.query_vectors[::-1].copy(),
    )
    return Gallery(
        items=g.items[::-1],
        queries=g.queries[::-1],
        ocr=dict(reversed(g.ocr.items())),
        embeddings=emb,
    )


def _renamed(g: Gallery, mapping: dict[str, str]) -> Gallery:
    def new(image_id: str) -> str:
        prefix, _, rest = image_id.partition("-")
        return f"{mapping[prefix]}-{rest}"

    return Gallery(
        items=tuple(replace(i, image_id=new(i.image_id)) for i in g.items),
        queries=tuple(replace(q, qid=q.qid, image_id=new(q.image_id)) for q in g.queries),
        ocr={new(k): replace(v, image_id=new(k)) for k, v in g.ocr.items()},
        embeddings=g.embeddings,
    )


def test_tie_values_are_deterministic_and_differ_per_query():
    tb = TieBreaker(["a", "b", "c", "d"])
    assert np.array_equal(tb.for_query("q1"), tb.for_query("q1"))
    assert not np.array_equal(tb.for_query("q1"), tb.for_query("q2"))


def test_file_order_does_not_change_any_ranking(small_filler):
    a, b = small_filler, _reordered(small_filler)
    ra, rb = Retriever.from_gallery(a), Retriever.from_gallery(b)
    for arm in ("clip", "ocr_bm25", *FUSION_ARMS):
        assert rank_queries(ra, a, arm, weight=0.4) == rank_queries(rb, b, arm, weight=0.4), arm


def test_file_order_does_not_change_the_study(small_filler):
    config = EvalConfig(resamples=50, split="all")
    a = run_study(small_filler, config)
    b = run_study(_reordered(small_filler), config)
    sweep_a = [r["dev_ndcg@10"] for r in a.config["weight"]["sweep"]]
    sweep_b = [r["dev_ndcg@10"] for r in b.config["weight"]["sweep"]]
    assert sweep_a == pytest.approx(sweep_b, abs=1e-12)  # summation order only
    assert a.config["weight"]["chosen_clip_weight"] == b.config["weight"]["chosen_clip_weight"]

    def means(result):
        return {
            (x["kind"], x["slice"], x["metric"], x.get("arm"), x.get("arm_a"), x.get("arm_b")): x[
                "mean"
            ]
            for x in result.rows
        }

    ma, mb = means(a), means(b)
    assert ma.keys() == mb.keys()
    assert [ma[k] for k in ma] == pytest.approx([mb[k] for k in ma], abs=1e-12)


def test_swapping_which_source_sorts_first_has_no_systematic_effect(small_filler):
    """Rename ids so textcaps sorts before coco instead of after: results move only by noise."""
    swapped = _renamed(small_filler, {"coco": "zz", "textcaps": "aa"})
    scores = {}
    for name, gallery in (("orig", small_filler), ("swapped", swapped)):
        retriever = Retriever.from_gallery(gallery)
        for arm in ("fuse_rrf", "fuse_score"):
            ranks = rank_queries(retriever, gallery, arm)
            matrix = metrics_matrix(ranks, gallery.queries)
            for source in ("coco", "textcaps"):
                mask = np.array([q.source == source for q in gallery.queries])
                scores[name, arm, source] = matrix[mask, NDCG].mean()
    for arm in ("fuse_rrf", "fuse_score"):
        for source in ("coco", "textcaps"):
            shift = scores["swapped", arm, source] - scores["orig", arm, source]
            assert abs(shift) < 0.04, (arm, source, shift)


def test_cross_source_rrf_tie_is_not_won_by_one_source():
    """An image only CLIP found at rank 1 ties exactly with one only OCR found at rank 1."""
    ids = ["coco-1", "tc-1"]
    tb = TieBreaker(ids)
    coco_first = 0
    n = 400
    for n_query in range(n):
        tie = tb.for_query(f"q{n_query}")
        ties = {i: int(tie[k]) for k, i in enumerate(ids)}
        fused = fuse_rankings("fuse_rrf", [("coco-1", 0.9)], [("tc-1", 7.0)], top_k=10, ties=ties)
        assert fused[0][1] == fused[1][1]  # a genuine exact tie
        coco_first += fused[0][0] == "coco-1"
    assert 0.4 * n < coco_first < 0.6 * n

    # Without per-query ties the vendored id order always lets coco win: the bias being removed.
    plain = fuse_rankings("fuse_rrf", [("coco-1", 0.9)], [("tc-1", 7.0)], top_k=10)
    assert plain[0][0] == "coco-1"


def test_equal_weight_score_tie_is_not_won_by_one_source():
    ids = ["coco-1", "tc-1"]
    tb = TieBreaker(ids)
    wins = 0
    for n_query in range(400):
        tie = tb.for_query(f"q{n_query}")
        ties = {i: int(tie[k]) for k, i in enumerate(ids)}
        # Both top-1s are absent from the other list: both normalise to 1.0 * 0.5.
        fused = fuse_rankings(
            "fuse_score",
            [("coco-1", 0.9), ("x", 0.1)],
            [("tc-1", 7.0), ("y", 1.0)],
            top_k=10,
            ties=ties,
        )
        wins += fused[0][0] == "coco-1"
    assert 160 < wins < 240


def _identical_ocr_gallery(n: int = 30) -> Gallery:
    items = tuple(
        GalleryItem(
            f"{'coco' if i < n // 2 else 'textcaps'}-{i:03d}", "coco" if i < n // 2 else "textcaps"
        )
        for i in range(n)
    )
    ocr = {i.image_id: OcrRecord(i.image_id, (OcrLine("stop sign", 0.9),)) for i in items}
    return Gallery(items=items, queries=(), ocr=ocr)


def test_bm25_ties_at_the_cutoff_are_resolved_by_the_hash_not_file_order():
    gallery = _identical_ocr_gallery()
    index = MultimodalIndex.from_gallery(gallery, need_dense=False)
    n_queries, k = 600, 5
    appearances = dict.fromkeys(gallery.image_ids, 0)
    rank1 = dict.fromkeys(gallery.image_ids, 0)
    for n in range(n_queries):
        hits = index.ocr_search("stop", k, f"q{n}")
        assert len(hits) == k
        for image_id, _ in hits:
            appearances[image_id] += 1
        rank1[hits[0][0]] += 1
    expected = n_queries * k / len(appearances)  # 100 each if every image is equally likely
    assert all(0.55 * expected < c < 1.45 * expected for c in appearances.values())
    first_half = sum(c for i, c in appearances.items() if i.startswith("coco"))
    second_half = sum(c for i, c in appearances.items() if i.startswith("textcaps"))
    assert abs(first_half - second_half) < 0.1 * n_queries * k


def test_gold_is_not_systematically_favoured_across_a_cross_source_bm25_tie():
    """Identical OCR everywhere: the gold's average rank is the same for coco and textcaps."""
    gallery = _identical_ocr_gallery()
    index = MultimodalIndex.from_gallery(gallery, need_dense=False)
    coco_gold, tc_gold = "coco-000", "textcaps-029"
    mean_rank = {}
    for gold in (coco_gold, tc_gold):
        ranks = []
        for n in range(400):
            ids = [i for i, _ in index.ocr_search("stop", 30, f"{gold}:{n}")]
            ranks.append(ids.index(gold) + 1)
        mean_rank[gold] = float(np.mean(ranks))
    assert abs(mean_rank[coco_gold] - mean_rank[tc_gold]) < 1.5
    assert abs(mean_rank[coco_gold] - 15.5) < 1.5
