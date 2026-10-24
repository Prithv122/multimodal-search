"""What the four fixtures say about the pre-registered checks.

``planted`` and ``null`` show the pipeline detects an effect and stays quiet without one.
``filler`` and ``shuffled`` are the harder cases: OCR text that overlaps captions through common
words, and OCR that is confidently wrong. Their results are recorded here as found; the fixtures
were not adjusted to make a check pass.
"""

from __future__ import annotations

import pytest

from mmsearch.arms import FUSION_ARMS, Retriever
from mmsearch.evaluate import EvalConfig, preregistered_checks, run_study

TEST = EvalConfig(resamples=1000, split="test")


def _diff(result, slice_name, arm_a, arm_b="clip"):
    rows = [
        r
        for r in result.rows
        if r["kind"] == "diff"
        and r["slice"] == slice_name
        and r["metric"] == "ndcg@10"
        and (r["arm_a"], r["arm_b"]) == (arm_a, arm_b)
    ]
    assert len(rows) == 1
    return rows[0]


@pytest.fixture(scope="module")
def filler_result(filler):
    return run_study(filler, TEST)


@pytest.fixture(scope="module")
def shuffled_result(shuffled):
    return run_study(shuffled, TEST)


def test_filler_tokens_really_overlap_captions_and_ocr(filler):
    coco_query = next(q for q in filler.queries if q.source == "coco")
    words = set(coco_query.text.split())
    ocr_words = {w for rec in filler.ocr.values() for ln in rec.lines for w in ln.text.split()}
    assert words & ocr_words & {"a", "the", "of", "on", "in", "with"}


def test_planted_effect_survives_filler_on_the_reading_slice(filler_result):
    h1 = filler_result.checks["H1_reading"]
    assert h1["status"] == "PASS"
    assert h1["ci_lo"] > 0.0
    for arm in FUSION_ARMS:
        assert _diff(filler_result, "source=textcaps", arm)["ci_lo"] > 0.0, arm


def test_filler_makes_fusion_cost_the_visual_slice(filler_result):
    """Finding, not a tuned outcome: with shared filler words the weak top OCR hit is scaled
    to 1.0 by min-max and pulls wrong images up. Unweighted fusions lose a lot on the visual
    slice, and even the dev-weighted arm fails the 0.02 tolerance (H2)."""
    for arm in ("fuse_rrf", "fuse_score"):
        assert _diff(filler_result, "source=coco", arm)["ci_hi"] < -0.02, arm
    best = filler_result.checks["best_fusion"]
    assert _diff(filler_result, "source=coco", best)["mean"] < 0.0
    assert filler_result.checks["H2_visual"]["status"] == "FAIL"


def test_shuffled_ocr_gives_no_reading_gain(shuffled_result):
    assert shuffled_result.checks["H1_reading"]["status"] == "FAIL"
    for arm in FUSION_ARMS:
        assert not _diff(shuffled_result, "source=textcaps", arm)["ci_lo"] > 0.0, arm


def test_shuffled_ocr_is_confidently_wrong(shuffled):
    """BM25 puts a *different* image first for a reading query, and does so with a high score."""
    retriever = Retriever.from_gallery(shuffled)
    queries = [q for q in shuffled.queries if q.source == "textcaps"][:200]
    wrong_first = 0
    for q in queries:
        hits = retriever.rank("ocr_bm25", q.text, query_key=q.qid)
        wrong_first += bool(hits) and hits[0][0] != q.image_id
    assert wrong_first > 0.8 * len(queries)


def test_dev_selection_protects_the_visual_slice_when_ocr_is_wrong(shuffled_result):
    """With a free weight the dev-tuned arm retreats to CLIP (w = 1), so H2 holds; H1 fails."""
    assert shuffled_result.config["weight"]["chosen_clip_weight"] == 1.0
    assert shuffled_result.checks["best_fusion"] == "fuse_score_weighted"
    assert shuffled_result.checks["H2_visual"]["status"] == "PASS"


def test_a_deliberately_harmful_fusion_fails_h2_on_shuffled_ocr(shuffled):
    """Fix the weight at 0.5 so no arm can escape to CLIP: the harm must be caught by H2."""
    result = run_study(shuffled, EvalConfig(resamples=1000, split="test", weight_grid=(0.5,)))
    assert result.checks["H2_visual"]["status"] == "FAIL"
    assert result.checks["H2_visual"]["diff_ndcg@10"] < -0.1
    assert result.checks["H1_reading"]["status"] == "FAIL"


def test_forcing_the_harmful_arm_as_best_fusion_fails_h2(shuffled_result):
    checks = preregistered_checks(shuffled_result.rows, "fuse_rrf", {}, TEST)
    assert checks["H2_visual"]["status"] == "FAIL"
    assert checks["H2_visual"]["ci_hi"] < -0.02


def test_best_fusion_selection_is_recorded_with_the_dev_advantage(filler_result):
    selection = filler_result.checks["best_fusion_selection"]
    assert selection["chosen"] == filler_result.checks["best_fusion"]
    assert set(selection["dev_ndcg@10"]) == set(FUSION_ARMS)
    assert selection["chosen_clip_weight"] == filler_result.config["weight"]["chosen_clip_weight"]
    assert "tuned on dev" in selection["note"]
