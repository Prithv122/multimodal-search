"""Bootstrap, dev-only selection, slices, and detection of a planted effect."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import mmsearch.evaluate as ev
from mmsearch.arms import ARMS, FUSION_ARMS
from mmsearch.evaluate import (
    METRICS,
    EvalConfig,
    cluster_bootstrap_means,
    pick_best_fusion,
    run_study,
    select_weight,
)
from mmsearch.gallery import SOURCES, Gallery, Query
from mmsearch.split import split_images

FAST = EvalConfig(resamples=1000)


def _row(result, kind, slice_name, metric="ndcg@10", **match):
    rows = [
        r
        for r in result.rows
        if r["kind"] == kind
        and r["slice"] == slice_name
        and r["metric"] == metric
        and all(r.get(k) == v for k, v in match.items())
    ]
    assert len(rows) == 1, (kind, slice_name, metric, match, len(rows))
    return rows[0]


@pytest.fixture(scope="module")
def planted_result(planted):
    return run_study(planted, FAST)


@pytest.fixture(scope="module")
def null_result(null):
    return run_study(null, FAST)


# -- cluster bootstrap --------------------------------------------------------
def test_bootstrap_resamples_images_not_queries():
    # Image 0 has five captions that all score 1; image 1 has one caption that scores 0.
    values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    clusters = np.array([0, 0, 0, 0, 0, 1])
    means = cluster_bootstrap_means(values, clusters, resamples=3000, seed=1)[:, 0]
    # Drawing 2 images with replacement can only give: (0,0) -> 1, (0,1) -> 5/6, (1,1) -> 0.
    # A query-level bootstrap would also produce 4/6, 3/6, 2/6 and so on.
    assert set(np.round(means, 12)) == {0.0, round(5 / 6, 12), 1.0}


def test_captions_of_one_image_are_not_independent_draws():
    rng = np.random.default_rng(0)
    per_image = rng.random(40)
    one_each = cluster_bootstrap_means(per_image, np.arange(40), resamples=500, seed=3)
    five_each = cluster_bootstrap_means(
        np.repeat(per_image, 5), np.repeat(np.arange(40), 5), resamples=500, seed=3
    )
    # Five identical captions per image add no information: same intervals as one each.
    np.testing.assert_allclose(one_each, five_each)


def test_bootstrap_is_seeded_and_centred_on_the_sample_mean():
    rng = np.random.default_rng(1)
    values = rng.random(200)
    clusters = np.repeat(np.arange(50), 4)
    a = cluster_bootstrap_means(values, clusters, resamples=4000, seed=7)
    b = cluster_bootstrap_means(values, clusters, resamples=4000, seed=7)
    c = cluster_bootstrap_means(values, clusters, resamples=4000, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert abs(a.mean() - values.mean()) < 0.01


def test_columns_share_resamples_so_differences_are_paired():
    rng = np.random.default_rng(2)
    base = rng.random(60)
    values = np.stack([base, base + 0.25], axis=1)
    means = cluster_bootstrap_means(values, np.repeat(np.arange(20), 3), resamples=300, seed=5)
    np.testing.assert_allclose(means[:, 1] - means[:, 0], 0.25)


# -- weight selection on dev only -----------------------------------------------
def _scramble_test_labels(gallery: Gallery) -> Gallery:
    """Point every test-split query at a different *test* image of the same source."""
    assignment = split_images(gallery.items)
    by_source = {
        s: [i.image_id for i in gallery.items if i.source == s and assignment[i.image_id] == "test"]
        for s in SOURCES
    }
    queries = []
    for q in gallery.queries:
        if assignment[q.image_id] == "test":
            ids = by_source[q.source]
            q = replace(q, image_id=ids[(ids.index(q.image_id) + 7) % len(ids)])
        queries.append(q)
    return Gallery(
        items=gallery.items, queries=tuple(queries), ocr=gallery.ocr, embeddings=gallery.embeddings
    )


def test_weight_selection_never_sees_test_queries(small, monkeypatch):
    seen: list[set[str]] = []
    real = ev.select_weight

    def spy(dev_queries, base_rankings, **kwargs):
        seen.append({q.qid for q in dev_queries} | set(base_rankings))
        return real(dev_queries, base_rankings, **kwargs)

    monkeypatch.setattr(ev, "select_weight", spy)
    run_study(small, EvalConfig(resamples=50))

    assignment = split_images(small.items)
    dev = {q.qid for q in small.queries if assignment[q.image_id] == "dev"}
    test = {q.qid for q in small.queries if assignment[q.image_id] == "test"}
    assert len(seen) == 1
    assert seen[0] == dev
    assert not seen[0] & test


def test_dev_choices_do_not_change_when_test_labels_change(small):
    scrambled = _scramble_test_labels(small)
    a = run_study(small, EvalConfig(resamples=200))
    b = run_study(scrambled, EvalConfig(resamples=200))

    assert a.config["weight"]["sweep"] == b.config["weight"]["sweep"]
    assert a.config["weight"]["chosen_clip_weight"] == b.config["weight"]["chosen_clip_weight"]
    assert a.checks["best_fusion"] == b.checks["best_fusion"]
    assert a.checks["best_fusion_dev_ndcg@10"] == b.checks["best_fusion_dev_ndcg@10"]
    # ... while the test-split numbers did move, so the scramble was not a no-op.
    arm_a = _row(a, "arm", "all", arm="clip")["mean"]
    arm_b = _row(b, "arm", "all", arm="clip")["mean"]
    assert arm_a != pytest.approx(arm_b, abs=0.05)


def test_weight_ties_go_to_the_weight_closest_to_half():
    q = Query("q", "x", "gold", "coco")
    base = {"q": ([("gold", 0.9), ("other", 0.1)], [])}  # OCR silent: every w > 0 ranks gold first
    choice = select_weight([q], base, top_k=10)
    assert choice.weight == 0.5
    assert len(choice.table) == 21
    assert choice.table[0]["dev_ndcg@10"] == 0.0  # w = 0 discards CLIP entirely


def test_weight_selection_needs_dev_queries():
    with pytest.raises(ValueError, match="dev queries"):
        select_weight([], {}, top_k=10)


def test_best_fusion_is_the_highest_dev_arm_with_stable_ties():
    assert pick_best_fusion({"fuse_rrf": 0.5, "fuse_score": 0.7, "fuse_score_weighted": 0.6}) == (
        "fuse_score"
    )
    assert pick_best_fusion({a: 0.5 for a in FUSION_ARMS}) == FUSION_ARMS[0]


def test_weight_grid_is_the_registered_one(planted_result):
    grid = [row["weight"] for row in planted_result.config["weight"]["sweep"]]
    assert grid == [round(i * 0.05, 2) for i in range(21)]


# -- planted effect vs null ---------------------------------------------------------
def test_planted_effect_is_detected_on_the_reading_slice(planted_result):
    diff = _row(
        planted_result,
        "diff",
        "source=textcaps",
        arm_a=planted_result.checks["best_fusion"],
        arm_b="clip",
    )
    assert diff["mean"] > 0.3
    assert diff["ci_lo"] > 0.0
    h1 = planted_result.checks["H1_reading"]
    assert h1["status"] == "PASS"
    assert h1["ci_excludes_zero"] and h1["gain_ok"]


def test_every_fusion_arm_beats_clip_on_the_planted_reading_slice(planted_result):
    for arm in FUSION_ARMS:
        diff = _row(planted_result, "diff", "source=textcaps", arm_a=arm, arm_b="clip")
        assert diff["ci_lo"] > 0.0, arm


def test_planted_visual_slice_is_unharmed(planted_result):
    h2 = planted_result.checks["H2_visual"]
    assert h2["status"] == "PASS"
    assert abs(h2["diff_ndcg@10"]) <= 0.02


def test_ocr_alone_loses_on_visual_and_clip_loses_on_reading(planted_result):
    visual = _row(planted_result, "diff", "source=coco", arm_a="ocr_bm25", arm_b="clip")
    reading = _row(planted_result, "diff", "source=textcaps", arm_a="ocr_bm25", arm_b="clip")
    assert visual["ci_hi"] < 0.0
    assert reading["ci_lo"] > 0.0


def test_null_fixture_does_not_trigger_the_effect(null_result):
    assert null_result.checks["H1_reading"]["status"] == "FAIL"
    for arm in FUSION_ARMS:
        diff = _row(null_result, "diff", "source=textcaps", arm_a=arm, arm_b="clip")
        assert not diff["ci_lo"] > 0.0, arm  # no fusion arm shows a significant gain
        assert diff["mean"] < 0.05, arm


def test_null_fixture_still_passes_the_visual_check(null_result):
    assert null_result.checks["H2_visual"]["status"] == "PASS"


# -- structure of the result ----------------------------------------------------------
def test_rows_cover_arms_slices_metrics_and_pairs(planted_result):
    arm_rows = [r for r in planted_result.rows if r["kind"] == "arm"]
    slices = {r["slice"] for r in arm_rows}
    assert slices == {
        "all",
        "source=coco",
        "source=textcaps",
        "tag=visual",
        "tag=text",
        "tag=both",
    }
    assert {r["arm"] for r in arm_rows} == set(ARMS)
    assert {r["metric"] for r in arm_rows} == set(METRICS)
    assert len(arm_rows) == len(slices) * len(ARMS) * len(METRICS)
    for r in planted_result.rows:
        assert r["ci_lo"] <= r["mean"] + 1e-9 <= r["ci_hi"] + 2e-9
        assert r["n_images"] > 0 and r["n_queries"] >= r["n_images"]


def test_diff_mean_is_the_difference_of_arm_means(planted_result):
    diff = _row(planted_result, "diff", "all", arm_a="fuse_rrf", arm_b="clip")
    a = _row(planted_result, "arm", "all", arm="fuse_rrf")["mean"]
    b = _row(planted_result, "arm", "all", arm="clip")["mean"]
    assert diff["mean"] == pytest.approx(a - b)


def test_result_is_json_serialisable(planted_result):
    payload = json.loads(json.dumps(planted_result.to_json()))
    assert set(payload) == {"config", "rows", "checks"}
    assert payload["config"]["split"] == "test"


def test_tag_slices_only_appear_when_tags_exist(small):
    untagged = Gallery(
        items=small.items,
        queries=tuple(replace(q, tag=None) for q in small.queries),
        ocr=small.ocr,
        embeddings=small.embeddings,
    )
    result = run_study(untagged, EvalConfig(resamples=50))
    assert not any(r["slice"].startswith("tag=") for r in result.rows)


def test_split_selects_the_images_scored(small):
    n_dev = run_study(small, EvalConfig(split="dev", resamples=50))
    n_test = run_study(small, EvalConfig(split="test", resamples=50))
    n_all = run_study(small, EvalConfig(split="all", resamples=50))
    images = {
        name: _row(r, "arm", "all", arm="clip")["n_images"]
        for name, r in (("dev", n_dev), ("test", n_test), ("all", n_all))
    }
    assert images == {"dev": 16, "test": 64, "all": 80}
    assert n_test.checks["confirmatory"] and not n_dev.checks["confirmatory"]
    assert not n_all.checks["confirmatory"]


def test_arm_filter_only_changes_rows_not_checks(small):
    full = run_study(small, EvalConfig(resamples=100))
    filtered = run_study(small, EvalConfig(resamples=100, arms=("clip", "fuse_rrf")))
    assert {r["arm"] for r in filtered.rows if r["kind"] == "arm"} == {"clip", "fuse_rrf"}
    assert {(r["arm_a"], r["arm_b"]) for r in filtered.rows if r["kind"] == "diff"} == {
        ("fuse_rrf", "clip")
    }
    assert filtered.checks == full.checks


def test_thresholds_come_from_config(small):
    strict = run_study(small, EvalConfig(resamples=200, reading_min_gain=0.99))
    assert strict.checks["H1_reading"]["status"] == "FAIL"
    assert strict.checks["H1_reading"]["min_gain"] == 0.99


def test_unknown_split_or_arm_rejected(small):
    with pytest.raises(ValueError, match="split"):
        run_study(small, EvalConfig(split="validation"))
    with pytest.raises(ValueError, match="unknown arms"):
        run_study(small, EvalConfig(arms=("nope",)))
