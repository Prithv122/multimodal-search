"""The study: per-query metrics, dev-only weight selection, cluster bootstrap, slice tables.

Pipeline (:func:`run_study`)
    1. Split images into dev / test (``split.py``). Queries follow their image.
    2. Rank every query with the CLIP and OCR arms once (top ``top_k``), then derive the fusion
       arms from those two lists.
    3. **Choose the CLIP weight on dev only** (:func:`select_weight`): sweep ``w`` over a fixed
       grid (0.00 to 1.00 in steps of 0.05), fuse with per-query min-max normalised scores
       (see ``arms.py``), and keep the ``w`` with the highest mean nDCG@10 over *all dev
       queries pooled* (both slices). Ties go to the weight closest to 0.5, then the smaller
       weight. Only dev queries and their labels are passed in; test labels are never read.
    4. **Choose "best fusion" on dev only**: the fusion arm with the highest dev nDCG@10
       (all dev queries pooled; ties by the order ``fuse_rrf``, ``fuse_score``,
       ``fuse_score_weighted``). This is the single arm the pre-registered tests are about, so
       the test comparison is not a best-of-three pick.
    5. Score the requested split per slice with a **cluster bootstrap over images**: images are
       resampled with replacement and each carries all its captions in that slice. Every arm
       is scored on the *same* resample, so paired differences between arms are exact: the
       interval of ``fuse - clip`` is the interval of the difference, not a comparison of two
       overlapping intervals.
    6. Evaluate the two pre-registered quantities and report PASS / FAIL.

Slices are ``all``; ``source=coco`` (the visual slice) and ``source=textcaps`` (the reading
slice); and, when any query carries a hand tag, ``tag=visual``, ``tag=text`` and ``tag=both``.
The source slice is the dataset, not the query type; the tag slices are the robustness check.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from mmsearch._vendor.metrics import RetrievalScore
from mmsearch.arms import ARMS, FUSION_ARMS, BaseRankings, Ranking, Retriever, fuse_rankings
from mmsearch.gallery import TAGS, Gallery, Query
from mmsearch.split import DEFAULT_DEV_FRACTION, DEFAULT_SPLIT_SEED, split_images

# MRR is computed over each arm's top-``top_k`` list (100 by default), so a gold image outside it
# scores 0: that is why the metric is named MRR@100. The vendored function calls it "mrr".
METRICS = ("recall@1", "recall@5", "recall@10", "ndcg@10", "mrr@100")
_VENDORED_KEY = {"mrr@100": "mrr"}
NDCG = METRICS.index("ndcg@10")
SPLITS = ("dev", "test", "all")
DEFAULT_WEIGHT_GRID = tuple(round(i * 0.05, 2) for i in range(21))
READING_SLICE = "source=textcaps"
VISUAL_SLICE = "source=coco"

# Pairs (a, b) for which a - b is reported on the shared resamples.
DIFF_PAIRS = (
    *((f, "clip") for f in FUSION_ARMS),
    *((f, "ocr_bm25") for f in FUSION_ARMS),
    ("ocr_bm25", "clip"),
)


@dataclass(frozen=True)
class EvalConfig:
    split: str = "dev"  # test is scored only when asked for explicitly
    arms: tuple[str, ...] | None = None  # row filter only; every arm is always computed
    resamples: int = 20_000
    seed: int = 47  # bootstrap seed
    split_seed: int = DEFAULT_SPLIT_SEED
    dev_fraction: float = DEFAULT_DEV_FRACTION
    top_k: int = 100
    min_confidence: float | None = None
    weight_grid: tuple[float, ...] = DEFAULT_WEIGHT_GRID
    reading_min_gain: float = 0.10  # H1: best fusion - clip >= this on the reading slice
    visual_max_abs_diff: float = 0.02  # H2: |best fusion - clip| <= this on the visual slice
    alpha: float = 0.05


@dataclass
class WeightChoice:
    weight: float
    objective: float
    table: list[dict[str, float]]


@dataclass
class StudyResult:
    config: dict[str, Any]
    rows: list[dict[str, Any]]
    checks: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"config": self.config, "rows": self.rows, "checks": self.checks}


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------
def query_metrics(ranking: Ranking, gold_image_id: str) -> np.ndarray:
    """The five metrics for one query whose only relevant item is ``gold_image_id``."""
    score = RetrievalScore.evaluate([image_id for image_id, _ in ranking], [gold_image_id])
    values = score.as_dict()
    return np.array([values[_VENDORED_KEY.get(m, m)] for m in METRICS], dtype=np.float64)


def metrics_matrix(rankings: Mapping[str, Ranking], queries: Sequence[Query]) -> np.ndarray:
    """``(n_queries, 5)`` metrics in the order of ``queries``."""
    return np.array([query_metrics(rankings[q.qid], q.image_id) for q in queries])


# ---------------------------------------------------------------------------
# Weight selection (dev only)
# ---------------------------------------------------------------------------
def select_weight(
    dev_queries: Sequence[Query],
    base_rankings: Mapping[str, BaseRankings],
    *,
    top_k: int,
    grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> WeightChoice:
    """Pick the CLIP weight maximising pooled mean nDCG@10 over ``dev_queries``.

    This function sees only the queries handed to it; the caller passes dev queries.
    """
    if not dev_queries:
        raise ValueError("cannot select a weight without dev queries")
    table: list[dict[str, float]] = []
    for w in grid:
        total = 0.0
        for q in dev_queries:
            base = base_rankings[q.qid]
            fused = fuse_rankings(
                "fuse_score_weighted", base.clip, base.ocr, top_k=top_k, weight=w, ties=base.ties
            )
            total += float(query_metrics(fused, q.image_id)[NDCG])
        table.append({"weight": float(w), "dev_ndcg@10": total / len(dev_queries)})
    best = min(
        table, key=lambda r: (-round(r["dev_ndcg@10"], 12), abs(r["weight"] - 0.5), r["weight"])
    )
    return WeightChoice(best["weight"], best["dev_ndcg@10"], table)


def pick_best_fusion(dev_ndcg: Mapping[str, float]) -> str:
    """Fusion arm with the highest dev nDCG@10; ties by ``FUSION_ARMS`` order."""
    return max(FUSION_ARMS, key=lambda a: (round(dev_ndcg[a], 12), -FUSION_ARMS.index(a)))


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap_means(
    values: np.ndarray, clusters: np.ndarray, *, resamples: int, seed: int | Sequence[int]
) -> np.ndarray:
    """Bootstrap means of ``values`` when whole clusters are resampled with replacement.

    ``values`` is ``(n_queries, n_cols)`` and ``clusters`` gives each query's image. Each
    resample draws as many images as there are distinct images, with replacement; an image
    brings all of its queries, so the resampled mean is ``sum(values of drawn images) /
    number of queries of drawn images``. Returns ``(resamples, n_cols)``. All columns share the
    same resamples, which is what makes differences between columns paired.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    _, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    sums = np.zeros((n_clusters, values.shape[1]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)

    rng = np.random.default_rng(seed)
    uniform = np.full(n_clusters, 1.0 / n_clusters)
    out = np.empty((resamples, values.shape[1]))
    chunk = max(1, 4_000_000 // n_clusters)
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        # Drawing n_clusters images with replacement is a multinomial over the images.
        draws = rng.multinomial(n_clusters, uniform, size=size).astype(np.float64)
        out[start : start + size] = (draws @ sums) / (draws @ counts)[:, None]
    return out


def percentile_interval(samples: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2], axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------
def _slices(queries: Sequence[Query]) -> list[tuple[str, np.ndarray]]:
    out = [("all", np.ones(len(queries), dtype=bool))]
    for source in ("coco", "textcaps"):
        out.append((f"source={source}", np.array([q.source == source for q in queries])))
    if any(q.tag for q in queries):
        for tag in TAGS:
            out.append((f"tag={tag}", np.array([q.tag == tag for q in queries])))
    return out


def run_study(gallery: Gallery, config: EvalConfig | None = None) -> StudyResult:
    config = config or EvalConfig()
    if config.split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {config.split!r}")
    unknown = set(config.arms or ()) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; expected {list(ARMS)}")

    queries = list(gallery.queries)
    assignment = split_images(
        gallery.items, dev_fraction=config.dev_fraction, seed=config.split_seed
    )
    query_split = np.array([assignment[q.image_id] for q in queries])
    dev_queries = [q for q, s in zip(queries, query_split, strict=True) if s == "dev"]
    if not dev_queries:
        raise ValueError("the dev split has no queries; cannot choose weight or best fusion")

    retriever = Retriever.from_gallery(
        gallery, top_k=config.top_k, min_confidence=config.min_confidence
    )
    base = {
        q.qid: retriever.base_rankings(q.text, gallery.query_vector(q.qid), q.qid) for q in queries
    }

    # -- dev-only choices ------------------------------------------------
    choice = select_weight(
        dev_queries,
        {q.qid: base[q.qid] for q in dev_queries},
        top_k=config.top_k,
        grid=config.weight_grid,
    )

    def arm_ranking(arm: str, qid: str) -> Ranking:
        b = base[qid]
        if arm == "clip":
            return b.clip
        if arm == "ocr_bm25":
            return b.ocr
        return fuse_rankings(
            arm, b.clip, b.ocr, top_k=config.top_k, weight=choice.weight, ties=b.ties
        )

    per_query = {
        arm: metrics_matrix({q.qid: arm_ranking(arm, q.qid) for q in queries}, queries)
        for arm in ARMS
    }
    dev_mask = query_split == "dev"
    dev_ndcg = {arm: float(per_query[arm][dev_mask, NDCG].mean()) for arm in ARMS}
    best_fusion = pick_best_fusion(dev_ndcg)

    # -- scoring the requested split ------------------------------------------
    eval_mask = (
        np.ones(len(queries), dtype=bool)
        if config.split == "all"
        else (query_split == config.split)
    )
    image_index = {item.image_id: i for i, item in enumerate(gallery.items)}
    clusters_all = np.array([image_index[q.image_id] for q in queries])

    rows: list[dict[str, Any]] = []
    best_single: dict[str, str] = {}
    for name, slice_mask in _slices(queries):
        take = slice_mask & eval_mask
        if not take.any():
            continue
        clusters = clusters_all[take]
        n_images, n_queries = int(np.unique(clusters).size), int(take.sum())
        block = np.stack([per_query[a][take] for a in ARMS], axis=1)  # (n, arms, metrics)
        point = block.mean(axis=0)
        boot = cluster_bootstrap_means(
            block.reshape(n_queries, -1),
            clusters,
            resamples=config.resamples,
            seed=[config.seed, zlib.crc32(name.encode())],
        ).reshape(config.resamples, len(ARMS), len(METRICS))

        dev_take = slice_mask & dev_mask
        if dev_take.any():
            best_single[name] = max(
                ("clip", "ocr_bm25"), key=lambda a: round(per_query[a][dev_take, NDCG].mean(), 12)
            )

        for a, arm in enumerate(ARMS):
            lo, hi = percentile_interval(boot[:, a, :], config.alpha)
            for m, metric in enumerate(METRICS):
                rows.append(
                    {
                        "kind": "arm",
                        "arm": arm,
                        "slice": name,
                        "metric": metric,
                        "mean": float(point[a, m]),
                        "ci_lo": float(lo[m]),
                        "ci_hi": float(hi[m]),
                        "n_images": n_images,
                        "n_queries": n_queries,
                    }
                )
        for arm_a, arm_b in DIFF_PAIRS:
            ia, ib = ARMS.index(arm_a), ARMS.index(arm_b)
            lo, hi = percentile_interval(boot[:, ia, :] - boot[:, ib, :], config.alpha)
            for m, metric in enumerate(METRICS):
                rows.append(
                    {
                        "kind": "diff",
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "slice": name,
                        "metric": metric,
                        "mean": float(point[ia, m] - point[ib, m]),
                        "ci_lo": float(lo[m]),
                        "ci_hi": float(hi[m]),
                        "n_images": n_images,
                        "n_queries": n_queries,
                    }
                )

    checks = preregistered_checks(rows, best_fusion, best_single, config)
    checks["best_fusion"] = best_fusion
    checks["best_fusion_dev_ndcg@10"] = {a: dev_ndcg[a] for a in FUSION_ARMS}
    checks["best_fusion_selection"] = {
        "chosen": best_fusion,
        "dev_ndcg@10": {a: dev_ndcg[a] for a in FUSION_ARMS},
        "chosen_clip_weight": choice.weight,
        "note": "fuse_score_weighted has its CLIP weight tuned on dev, the other two have no "
        "tuned parameter, so its dev score is in-sample and favours it in this pick",
    }
    checks["confirmatory"] = config.split == "test"

    if config.arms:
        keep = set(config.arms)
        rows = [
            r
            for r in rows
            if (r["kind"] == "arm" and r["arm"] in keep)
            or (r["kind"] == "diff" and r["arm_a"] in keep and r["arm_b"] in keep)
        ]

    n_dev = sum(1 for s in assignment.values() if s == "dev")
    result_config = {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()},
        "n_images": len(gallery.items),
        "n_queries": len(queries),
        "n_dev_images": n_dev,
        "n_test_images": len(assignment) - n_dev,
        "weight": {
            "chosen_clip_weight": choice.weight,
            "dev_objective": "mean ndcg@10 over all dev queries pooled",
            "dev_ndcg@10": choice.objective,
            "sweep": choice.table,
            "normalisation": "per-query min-max over each arm's top-k list",
        },
        "embedding_model": gallery.embeddings.model_name if gallery.embeddings else None,
        "fusion_arms": list(FUSION_ARMS),
    }
    return StudyResult(config=result_config, rows=rows, checks=checks)


# ---------------------------------------------------------------------------
# Pre-registered quantities
# ---------------------------------------------------------------------------
def _find(rows: Sequence[Mapping[str, Any]], slice_name: str, a: str, b: str) -> Mapping | None:
    for r in rows:
        if (
            r["kind"] == "diff"
            and r["slice"] == slice_name
            and r["arm_a"] == a
            and r["arm_b"] == b
            and r["metric"] == "ndcg@10"
        ):
            return r
    return None


def _diff_row(rows: Sequence[Mapping[str, Any]], slice_name: str, a: str, b: str) -> Mapping | None:
    if a == b:
        return None
    return _find(rows, slice_name, a, b)


def preregistered_checks(
    rows: Sequence[Mapping[str, Any]],
    best_fusion: str,
    best_single: Mapping[str, str],
    config: EvalConfig,
) -> dict[str, Any]:
    """H1 and H2 from the plan, plus the secondary "beats the best single arm" reading.

    H1 (reading slice): ``best_fusion - clip`` nDCG@10 >= ``reading_min_gain`` and its interval
    excludes 0 (lower bound > 0). H2 (visual slice): ``|best_fusion - clip|`` <=
    ``visual_max_abs_diff`` on the point estimate; the interval is reported alongside.
    """
    out: dict[str, Any] = {}

    row = _find(rows, READING_SLICE, best_fusion, "clip")
    if row is None:
        out["H1_reading"] = {"status": "N/A", "reason": "no reading-slice queries in this split"}
    else:
        gain_ok = row["mean"] >= config.reading_min_gain
        ci_ok = row["ci_lo"] > 0.0
        out["H1_reading"] = {
            "status": "PASS" if gain_ok and ci_ok else "FAIL",
            "slice": READING_SLICE,
            "best_fusion": best_fusion,
            "diff_ndcg@10": row["mean"],
            "ci_lo": row["ci_lo"],
            "ci_hi": row["ci_hi"],
            "min_gain": config.reading_min_gain,
            "gain_ok": gain_ok,
            "ci_excludes_zero": ci_ok,
        }

    row = _find(rows, VISUAL_SLICE, best_fusion, "clip")
    if row is None:
        out["H2_visual"] = {"status": "N/A", "reason": "no visual-slice queries in this split"}
    else:
        ok = abs(row["mean"]) <= config.visual_max_abs_diff
        out["H2_visual"] = {
            "status": "PASS" if ok else "FAIL",
            "slice": VISUAL_SLICE,
            "best_fusion": best_fusion,
            "diff_ndcg@10": row["mean"],
            "ci_lo": row["ci_lo"],
            "ci_hi": row["ci_hi"],
            "max_abs_diff": config.visual_max_abs_diff,
        }

    secondary: dict[str, Any] = {}
    for slice_name, single in best_single.items():
        row = _diff_row(rows, slice_name, best_fusion, single)
        if row is not None:
            secondary[slice_name] = {
                "best_single_arm_on_dev": single,
                "diff_ndcg@10": row["mean"],
                "ci_lo": row["ci_lo"],
                "ci_hi": row["ci_hi"],
                "fusion_beats_best_single": row["ci_lo"] > 0.0,
            }
    out["best_fusion_vs_best_single"] = secondary
    return out
