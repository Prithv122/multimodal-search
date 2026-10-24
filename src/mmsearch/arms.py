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
only fill the tail with alphabetical tie-breaks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mmsearch._vendor.fuse import fuse
from mmsearch.gallery import Gallery, Query
from mmsearch.index import MultimodalIndex

ARMS = ("clip", "ocr_bm25", "fuse_rrf", "fuse_score", "fuse_score_weighted")
FUSION_ARMS = ("fuse_rrf", "fuse_score", "fuse_score_weighted")
DEFAULT_TOP_K = 100

Ranking = list[tuple[str, float]]


def fuse_rankings(
    arm: str, clip_ranking: Ranking, ocr_ranking: Ranking, *, top_k: int, weight: float = 0.5
) -> Ranking:
    """Combine the two base rankings for one fusion arm. ``weight`` is the CLIP weight."""
    pair = [clip_ranking, ocr_ranking]
    if arm == "fuse_rrf":
        return fuse(pair, "rrf", k=top_k)
    if arm == "fuse_score":
        weight = 0.5
    elif arm != "fuse_score_weighted":
        raise ValueError(f"{arm!r} is not a fusion arm; expected one of {list(FUSION_ARMS)}")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    fused = fuse(pair, "score", k=top_k, weights=(weight, 1.0 - weight))
    return [(image_id, score) for image_id, score in fused if score > 1e-12]


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

    def base_rankings(self, text: str, vector: np.ndarray | None) -> tuple[Ranking, Ranking]:
        """The CLIP and OCR rankings every fusion is built from."""
        if vector is None:
            raise ValueError("this arm needs a query vector (use a stored caption or an encoder)")
        return self.index.clip_search(vector, self.top_k), self.index.ocr_search(text, self.top_k)

    def rank(
        self, arm: str, text: str, vector: np.ndarray | None = None, *, weight: float = 0.5
    ) -> Ranking:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
        if arm == "ocr_bm25":
            return self.index.ocr_search(text, self.top_k)
        if arm == "clip":
            if vector is None:
                raise ValueError("the clip arm needs a query vector")
            return self.index.clip_search(vector, self.top_k)
        clip_ranking, ocr_ranking = self.base_rankings(text, vector)
        return fuse_rankings(arm, clip_ranking, ocr_ranking, top_k=self.top_k, weight=weight)


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
        out[query.qid] = retriever.rank(arm, query.text, vector, weight=weight)
    return out
