"""The five arms: hand-computed fusion values and behaviour on the planted fixture."""

from __future__ import annotations

import numpy as np
import pytest

from mmsearch.arms import ARMS, Retriever, fuse_rankings, rank_queries

CLIP = [("a", 0.9), ("b", 0.5)]
OCR = [("b", 3.0), ("c", 1.0)]


def test_rrf_by_hand():
    fused = dict(fuse_rankings("fuse_rrf", CLIP, OCR, top_k=10))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)
    assert [i for i, _ in fuse_rankings("fuse_rrf", CLIP, OCR, top_k=10)] == ["b", "a", "c"]


def test_equal_weight_score_fusion_by_hand_and_drops_zero_evidence():
    # min-max: clip a=1, b=0; ocr b=1, c=0 -> a=.5, b=.5, c=0 (dropped: no positive evidence).
    fused = fuse_rankings("fuse_score", CLIP, OCR, top_k=10)
    assert fused == [("a", pytest.approx(0.5)), ("b", pytest.approx(0.5))]


def test_weighted_score_follows_the_clip_weight():
    high = fuse_rankings("fuse_score_weighted", CLIP, OCR, top_k=10, weight=0.8)
    assert [i for i, _ in high] == ["a", "b"]
    assert dict(high)["a"] == pytest.approx(0.8)
    assert dict(high)["b"] == pytest.approx(0.2)
    low = fuse_rankings("fuse_score_weighted", CLIP, OCR, top_k=10, weight=0.2)
    assert [i for i, _ in low] == ["b", "a"]


def test_weight_endpoints_use_only_one_arm():
    only_clip = fuse_rankings("fuse_score_weighted", CLIP, OCR, top_k=10, weight=1.0)
    assert [i for i, _ in only_clip] == ["a"]  # b scored 0 in clip's min-max, c gets nothing
    only_ocr = fuse_rankings("fuse_score_weighted", CLIP, OCR, top_k=10, weight=0.0)
    assert [i for i, _ in only_ocr] == ["b"]


def test_fusion_truncates_to_top_k():
    clip = [(f"i{n}", 1.0 - n / 100) for n in range(20)]
    assert len(fuse_rankings("fuse_rrf", clip, [], top_k=5)) == 5


def test_invalid_inputs():
    with pytest.raises(ValueError, match="not a fusion arm"):
        fuse_rankings("clip", CLIP, OCR, top_k=5)
    with pytest.raises(ValueError, match=r"weight must be in"):
        fuse_rankings("fuse_score_weighted", CLIP, OCR, top_k=5, weight=1.5)


def test_arms_on_planted_fixture(planted):
    retriever = Retriever.from_gallery(planted)
    coco = next(q for q in planted.queries if q.source == "coco")
    textcaps = next(q for q in planted.queries if q.source == "textcaps" and q.qid.endswith("q0"))

    # Visual content is solvable by CLIP and invisible to OCR.
    assert retriever.rank("clip", coco.text, planted.query_vector(coco.qid))[0][0] == coco.image_id
    assert retriever.rank("ocr_bm25", coco.text) == []

    # Printed words are found by OCR; CLIP cannot rely on them.
    ocr_hits = retriever.rank("ocr_bm25", textcaps.text)
    assert ocr_hits[0][0] == textcaps.image_id
    fused = retriever.rank("fuse_rrf", textcaps.text, planted.query_vector(textcaps.qid))
    assert fused[0][0] == textcaps.image_id


def test_every_arm_respects_top_k_and_is_sorted(planted):
    retriever = Retriever.from_gallery(planted, top_k=7)
    query = planted.queries[-1]
    for arm in ARMS:
        ranking = retriever.rank(arm, query.text, planted.query_vector(query.qid))
        assert len(ranking) <= 7
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)


def test_default_depth_is_100(planted):
    retriever = Retriever.from_gallery(planted)
    query = planted.queries[0]
    assert len(retriever.rank("clip", query.text, planted.query_vector(query.qid))) == 100


def test_clip_based_arms_need_a_vector(planted):
    retriever = Retriever.from_gallery(planted)
    for arm in ("clip", "fuse_rrf", "fuse_score", "fuse_score_weighted"):
        with pytest.raises(ValueError, match="query vector"):
            retriever.rank(arm, "v001")
    with pytest.raises(ValueError, match="unknown arm"):
        retriever.rank("nope", "v001")


def test_rank_queries_uses_stored_vectors(tiny):
    retriever = Retriever.from_gallery(tiny)
    out = rank_queries(retriever, tiny, "clip", tiny.queries[:3])
    assert list(out) == [q.qid for q in tiny.queries[:3]]
    expected = retriever.rank("clip", "", tiny.query_vector(tiny.queries[0].qid))
    assert out[tiny.queries[0].qid] == expected
    assert np.isfinite([s for _, s in expected]).all()
