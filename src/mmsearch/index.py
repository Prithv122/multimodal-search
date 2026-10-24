"""Wrappers that put image ids on the vendored dense and BM25 indexes.

**How OCR text becomes a BM25 document** (:func:`ocr_document`): the lines of an image are
kept in the order given, each line's text is whitespace-normalised, lines whose confidence is
below an optional floor are dropped, and the survivors are joined with a single space. There is
no floor by default (``min_confidence=None`` keeps every line, including low-confidence ones),
because the plan fixes OCR handling before the run; the floor exists as a parameter so a
sensitivity check is possible without code changes. An image with no lines gets the empty
document: it can never be returned by the ``ocr_bm25`` arm, which is the honest outcome for an
image with no readable text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mmsearch._vendor.bm25 import BM25Index
from mmsearch._vendor.dense import DenseIndex
from mmsearch.gallery import Gallery, OcrRecord


def ocr_document(record: OcrRecord | None, min_confidence: float | None = None) -> str:
    """Join an image's OCR lines into one BM25 document (see module docstring)."""
    if record is None:
        return ""
    parts = []
    for line in record.lines:
        if min_confidence is not None and line.confidence < min_confidence:
            continue
        text = " ".join(line.text.split())
        if text:
            parts.append(text)
    return " ".join(parts)


def build_dense_index(gallery: Gallery) -> DenseIndex:
    """Cosine index over the image vectors, keyed by image id."""
    emb = gallery.require_embeddings()
    return DenseIndex(gallery.image_ids, emb.image_vectors, model_name=emb.model_name)


def build_bm25_index(gallery: Gallery, min_confidence: float | None = None) -> BM25Index:
    """BM25 over one OCR document per image, keyed by image id."""
    ocr = gallery.require_ocr()
    ids = gallery.image_ids
    return BM25Index.build(ids, [ocr_document(ocr.get(i), min_confidence) for i in ids])


def _digest64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


class TieBreaker:
    """Per-query pseudo-random tie order that tracks neither source nor file order.

    Exact score ties are common (two BM25 documents with the same terms, an RRF pair found at
    the same rank by different arms). Breaking them by image id or by position in the file would
    favour whichever slice sorts first. Instead each (query, image) pair gets a 64-bit hash, and
    ties are ordered by it: deterministic, different for every query, and blind to the id's
    spelling order and the gallery file order.
    """

    def __init__(self, image_ids: Sequence[str]) -> None:
        self._base = np.array([_digest64("img:" + i) for i in image_ids], dtype=np.uint64)

    def for_query(self, key: str) -> np.ndarray:
        """``uint64`` tie value for every image, in gallery order (smaller sorts first)."""
        x = self._base ^ np.uint64(_digest64("query:" + key))
        with np.errstate(over="ignore"):  # splitmix64 finaliser: wrap-around is intended
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
        return x


def _top_by_score(
    ids: Sequence[str], scores: np.ndarray, tie: np.ndarray, k: int, *, positive_only: bool
) -> list[tuple[str, float]]:
    """Full sort by (score descending, tie value), then take ``k``.

    A full sort, not a partial selection, so tied items at the cut-off are all considered
    before the tie order decides which are kept.
    """
    if k <= 0:
        return []
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((tie, -scores))
    out: list[tuple[str, float]] = []
    for i in order:
        if positive_only and scores[i] <= 0.0:
            break
        out.append((ids[i], float(scores[i])))
        if len(out) == k:
            break
    return out


@dataclass
class MultimodalIndex:
    """The two retrieval indexes over the same image ids."""

    dense: DenseIndex | None
    bm25: BM25Index | None

    def __post_init__(self) -> None:
        ids = self.dense.chunk_ids if self.dense is not None else self.bm25.chunk_ids
        self.ids = list(ids)
        self.position = {image_id: i for i, image_id in enumerate(self.ids)}
        self.ties = TieBreaker(self.ids)

    def tie_values(self, query_key: str) -> np.ndarray:
        return self.ties.for_query(query_key)

    @classmethod
    def from_gallery(
        cls,
        gallery: Gallery,
        *,
        min_confidence: float | None = None,
        need_dense: bool = True,
        need_bm25: bool = True,
    ) -> MultimodalIndex:
        return cls(
            dense=build_dense_index(gallery) if need_dense else None,
            bm25=build_bm25_index(gallery, min_confidence) if need_bm25 else None,
        )

    def clip_search(
        self, vector: np.ndarray, k: int, query_key: str = "", *, tie: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        if self.dense is None:
            raise ValueError("no dense index was built")
        tie = self.tie_values(query_key) if tie is None else tie
        return _top_by_score(
            self.ids, self.dense.scores_for_vector(vector), tie, k, positive_only=False
        )

    def ocr_search(
        self, text: str, k: int, query_key: str = "", *, tie: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        """BM25 hits (positive scores only), ties ordered per query."""
        if self.bm25 is None:
            raise ValueError("no BM25 index was built")
        tie = self.tie_values(query_key) if tie is None else tie
        return _top_by_score(self.ids, self.bm25.scores(text), tie, k, positive_only=True)
