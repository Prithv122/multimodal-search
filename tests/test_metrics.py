"""Hand-computed metric values for a single relevant image."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmsearch.evaluate import METRICS, metrics_matrix, query_metrics
from mmsearch.gallery import Query


def _ranking(*ids: str) -> list[tuple[str, float]]:
    return [(i, 1.0 - n / 100) for n, i in enumerate(ids)]


def _as_dict(values: np.ndarray) -> dict[str, float]:
    return dict(zip(METRICS, values, strict=True))


def test_gold_at_rank_one():
    m = _as_dict(query_metrics(_ranking("g", "x", "y"), "g"))
    assert m == {"recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0, "ndcg@10": 1.0, "mrr@100": 1.0}


def test_gold_at_rank_three():
    m = _as_dict(query_metrics(_ranking("x", "y", "g", "z"), "g"))
    assert m["recall@1"] == 0.0
    assert m["recall@5"] == 1.0
    assert m["recall@10"] == 1.0
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(4))  # 0.5
    assert m["mrr@100"] == pytest.approx(1 / 3)


def test_gold_at_rank_seven():
    ids = [f"x{n}" for n in range(6)] + ["g"]
    m = _as_dict(query_metrics(_ranking(*ids), "g"))
    assert (m["recall@1"], m["recall@5"], m["recall@10"]) == (0.0, 0.0, 1.0)
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(8))  # 1/3
    assert m["mrr@100"] == pytest.approx(1 / 7)


def test_gold_beyond_ten_counts_only_for_mrr():
    ids = [f"x{n}" for n in range(11)] + ["g"]
    m = _as_dict(query_metrics(_ranking(*ids), "g"))
    assert (m["recall@10"], m["ndcg@10"]) == (0.0, 0.0)
    assert m["mrr@100"] == pytest.approx(1 / 12)


def test_gold_absent_or_empty_ranking():
    assert not query_metrics(_ranking("x", "y"), "g").any()
    assert not query_metrics([], "g").any()


def test_matrix_rows_follow_query_order():
    queries = [Query("q1", "t", "g1", "coco"), Query("q2", "t", "g2", "coco")]
    rankings = {"q1": _ranking("g1"), "q2": _ranking("x", "g2")}
    matrix = metrics_matrix(rankings, queries)
    assert matrix.shape == (2, len(METRICS))
    assert matrix[0, METRICS.index("mrr@100")] == 1.0
    assert matrix[1, METRICS.index("mrr@100")] == 0.5
