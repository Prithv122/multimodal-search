"""The five ranking arms.

Every arm maps one query to a ranked list of ``(image_id, score)``, best first, at most
``top_k`` long (default 100 so fusion has depth).

``clip``
    Cosine similarity between the caption embedding and every image embedding.
``ocr_bm25``
    BM25 of the caption text against each image's OCR document. Images sharing no term with
    the query score 0 and are absent, so the list can be shorter than ``top_k``.
``fuse_rrf``
    Reciprocal rank fusion of the two lists, constant k = 60 (the vendored default), equal
    weights.
``fuse_score``
    Equal-weight sum of per-query min-max normalised scores (``fuse_score_weighted`` with
    w = 0.5).
``fuse_score_weighted``
    ``w * norm(clip) + (1 - w) * norm(ocr_bm25)``.

**Score normalisation.** Min-max, per query and per arm, over that arm's own top-``top_k``
candidate list (the vendored ``score_fusion``). An image missing from one list contributes 0
for that arm. The same normalisation is used for the weight sweep, so the weight is chosen on
exactly the quantity it is later applied to. After fusion, images whose combined score is not
positive are dropped: they have no positive evidence from any arm (for instance the lowest
CLIP candidate, whose normalised score is 0, or either arm at weight 0), and keeping them would
only fill the tail with arbitrary ties.

**Ties.** Equal scores are ordered by a per-query hash of (query key, image id), never by image
id or file position (see :class:`mmsearch.index.TieBreaker`). This applies to both base lists
and to every fused list. The vendored fusion orders ties by image id, so it is asked for the
whole candidate union and the result is re-sorted here before truncating to ``top_k``.
Fused scores are compared after rounding to 12 decimals so that float noise cannot decide a
tie that is exact on paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from mmsearch._vendor.fuse import fuse
from mmsearch.gallery import Gallery, Query
from mmsearch.index import MultimodalIndex

ARMS = ("clip", "ocr_bm25", "fuse_rrf", "fuse_score", "fuse_score_weighted")
FUSION_ARMS = ("fuse_rrf", "fuse_score", "fuse_score_weighted")
DEFAULT_TOP_K = 100

Ranking = list[tuple[str, float]]


class BaseRankings(NamedTuple):
    """The two lists every fusion is built from, plus the tie values of their candidates."""

    clip: Ranking
    ocr: Ranking
    ties: Mapping[str, int] | None = None


def fuse_rankings(
    arm: str,
    clip_ranking: Ranking,
    ocr_ranking: Ranking,
    *,
    top_k: int,
    weight: float = 0.5,
    ties: Mapping[str, int] | None = None,
) -> Ranking:
    """Combine the two base rankings for one fusion arm. ``weight`` is the CLIP weight.

    ``ties`` maps each candidate image to its tie value; without it ties fall back to the
    vendored image-id order (only useful for hand-built examples).
    """
    pair = [clip_ranking, ocr_ranking]
    depth = max(1, len(clip_ranking) + len(ocr_ranking))  # whole union; truncate after re-sort
    if arm == "fuse_rrf":
        fused = fuse(pair, "rrf", k=depth)
    else:
        if arm == "fuse_score":
            weight = 0.5
        elif arm != "fuse_score_weighted":
            raise ValueError(f"{arm!r} is not a fusion arm; expected one of {list(FUSION_ARMS)}")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        fused = fuse(pair, "score", k=depth, weights=(weight, 1.0 - weight))
        fused = [(image_id, score) for image_id, score in fused if score > 1e-12]
    if ties is not None:
        fused.sort(key=lambda item: (-round(item[1], 12), ties[item[0]]))
    return fused[:top_k]


class Retriever:
    """Runs arms over a prebuilt :class:`MultimodalIndex`."""

    def __init__(self, index: MultimodalIndex, *, top_k: int = DEFAULT_TOP_K) -> None:
        self.index = index
        self.top_k = top_k

    @classmethod
    def from_gallery(
        cls, gallery: Gallery, *, top_k: int = DEFAULT_TOP_K, min_confidence: float | None = None
    ) -> Retriever:
        return cls(
            MultimodalIndex.from_gallery(gallery, min_confidence=min_confidence), top_k=top_k
        )

    def base_rankings(
        self, text: str, vector: np.ndarray | None, query_key: str | None = None
    ) -> BaseRankings:
        """The CLIP and OCR rankings every fusion is built from.

        ``query_key`` (the qid for stored captions; the text when omitted) seeds the tie order.
        """
        if vector is None:
            raise ValueError("this arm needs a query vector (use a stored caption or an encoder)")
        tie = self.index.tie_values(text if query_key is None else query_key)
        clip_ranking = self.index.clip_search(vector, self.top_k, tie=tie)
        ocr_ranking = self.index.ocr_search(text, self.top_k, tie=tie)
        position = self.index.position
        ties = {
            image_id: int(tie[position[image_id]]) for image_id, _ in (*clip_ranking, *ocr_ranking)
        }
        return BaseRankings(clip_ranking, ocr_ranking, ties)

    def rank(
        self,
        arm: str,
        text: str,
        vector: np.ndarray | None = None,
        *,
        weight: float = 0.5,
        query_key: str | None = None,
    ) -> Ranking:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
        key = text if query_key is None else query_key
        if arm == "ocr_bm25":
            return self.index.ocr_search(text, self.top_k, key)
        if arm == "clip":
            if vector is None:
                raise ValueError("the clip arm needs a query vector")
            return self.index.clip_search(vector, self.top_k, key)
        base = self.base_rankings(text, vector, key)
        return fuse_rankings(
            arm, base.clip, base.ocr, top_k=self.top_k, weight=weight, ties=base.ties
        )


def rank_queries(
    retriever: Retriever,
    gallery: Gallery,
    arm: str,
    queries: Sequence[Query] | None = None,
    *,
    weight: float = 0.5,
) -> dict[str, Ranking]:
    """Rank stored captions: ``qid -> ranking`` using the gallery's stored query vectors."""
    out: dict[str, Ranking] = {}
    for query in queries if queries is not None else gallery.queries:
        vector = gallery.query_vector(query.qid) if arm != "ocr_bm25" else None
        out[query.qid] = retriever.rank(arm, query.text, vector, weight=weight, query_key=query.qid)
    return out
