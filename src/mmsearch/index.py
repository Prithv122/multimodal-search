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


@dataclass
class MultimodalIndex:
    """The two retrieval indexes over the same image ids."""

    dense: DenseIndex | None
    bm25: BM25Index | None

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

    def clip_search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.dense is None:
            raise ValueError("no dense index was built")
        return self.dense.search_vector(vector, k)

    def ocr_search(self, text: str, k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise ValueError("no BM25 index was built")
        return self.bm25.search(text, k)
