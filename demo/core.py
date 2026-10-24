"""Demo logic that needs neither torch nor gradio: loading, ranking and result formatting.

The Gradio app (``demo/app.py``) is a thin layer over this module, so everything here can be
tested on synthetic data.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmsearch.arms import ARMS, FUSION_ARMS, Retriever
from mmsearch.gallery import (
    OCR_FILE,
    DataError,
    Gallery,
    OcrRecord,
    Query,
    load_embeddings,
    load_items,
    load_ocr,
    load_queries,
    validate_gallery,
)
from mmsearch.index import MultimodalIndex, ocr_document

DEFAULT_WEIGHT = 0.5
DEFAULT_TOP_K = 8
OCR_ARMS = ("ocr_bm25", *FUSION_ARMS)
OCR_SNIPPET_CHARS = 160


@dataclass(frozen=True)
class Hit:
    rank: int
    image_id: str
    score: float
    source: str
    url: str
    ocr_text: str | None  # None when OCR is not loaded; "" when the image has no readable text

    @property
    def caption(self) -> str:
        return f"#{self.rank} | {self.score:.3f} | {self.source} | {self.image_id}"


def load_demo_gallery(data_dir: str | Path = "data") -> tuple[Gallery, list[str]]:
    """Load the gallery for the demo, tolerating a missing or unfinished OCR file.

    Returns the gallery and a list of notices for the user. OCR is used only when the file
    covers every gallery image exactly; a partial file (the OCR run still in progress) would
    make "no text" indistinguishable from "not processed yet", so the demo drops it and runs
    CLIP-only with a notice instead of showing misleading BM25 results.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise DataError(f"data directory not found: {root}")
    items = load_items(root / "gallery.jsonl")
    queries = load_queries(root / "queries.jsonl")
    embeddings = load_embeddings(root, [i.image_id for i in items], [q.qid for q in queries])
    if embeddings is None:
        raise DataError(
            f"no embeddings under {root / 'embeddings'}; the demo needs the CLIP arrays"
        )
    notices: list[str] = []
    ocr: dict[str, OcrRecord] = {}
    ocr_path = root / OCR_FILE
    if not ocr_path.is_file():
        notices.append(
            f"OCR file {OCR_FILE} not found: running CLIP-only. "
            "The OCR and fusion arms fall back to the CLIP ranking."
        )
    else:
        records = load_ocr(ocr_path)
        wanted = {i.image_id for i in items}
        if set(records) == wanted:
            ocr = records
        else:
            notices.append(
                f"OCR file covers {len(set(records) & wanted)} of {len(wanted)} images (the OCR "
                "run is unfinished): running CLIP-only. The OCR and fusion arms fall back to "
                "the CLIP ranking."
            )
    gallery = Gallery(items=items, queries=queries, ocr=ocr, embeddings=embeddings)
    validate_gallery(gallery)
    return gallery, notices


def recorded_weight(results_dir: str | Path = "results/dev") -> tuple[float | None, str | None]:
    """The CLIP weight chosen on dev, read from a results file in ``results_dir``.

    Only the dev directory is read, never test results. Returns ``(weight, file name)`` or
    ``(None, None)`` when no file records one.
    """
    directory = Path(results_dir)
    if not directory.is_dir():
        return None, None
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        weight = _find_key(payload, "chosen_clip_weight")
        if (
            isinstance(weight, int | float)
            and not isinstance(weight, bool)
            and 0.0 <= float(weight) <= 1.0
        ):
            return float(weight), path.name
    return None, None


def _find_key(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        if key in node:
            return node[key]
        children: Sequence[Any] = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find_key(child, key)
        if found is not None:
            return found
    return None


def example_choices(
    queries: Sequence[Query], per_source: int = 4, sources: Sequence[str] = ("coco", "textcaps")
) -> list[tuple[str, str]]:
    """``(label, qid)`` pairs: ``per_source`` captions from each source, spread evenly."""
    out: list[tuple[str, str]] = []
    for source in sources:
        pool = [q for q in queries if q.source == source]
        if not pool:
            continue
        step = max(1, len(pool) // per_source)
        for query in pool[::step][:per_source]:
            text = query.text if len(query.text) <= 90 else query.text[:87] + "..."
            out.append((f"[{source}] {text}", query.qid))
    return out


def format_hits(
    ranking: Sequence[tuple[str, float]], gallery: Gallery, top_k: int = DEFAULT_TOP_K
) -> list[Hit]:
    """Turn a ranked ``(image_id, score)`` list into display rows (top ``top_k``)."""
    hits = []
    for rank, (image_id, score) in enumerate(ranking[: max(0, top_k)], start=1):
        item = gallery.item(image_id)
        ocr_text = ocr_document(gallery.ocr.get(image_id)) if gallery.has_ocr else None
        hits.append(Hit(rank, image_id, float(score), item.source, item.url, ocr_text))
    return hits


def gallery_items(hits: Sequence[Hit]) -> list[tuple[str, str]]:
    """``(image URL, caption)`` pairs for a gallery widget; hits without a URL are skipped."""
    return [(hit.url, hit.caption) for hit in hits if hit.url]


def ocr_rows(hits: Sequence[Hit]) -> list[list[Any]]:
    """Rows ``[rank, image_id, OCR text]`` for the side table."""
    rows = []
    for hit in hits:
        if hit.ocr_text is None:
            text = "(OCR not loaded)"
        elif not hit.ocr_text:
            text = "(no text found)"
        elif len(hit.ocr_text) > OCR_SNIPPET_CHARS:
            text = hit.ocr_text[: OCR_SNIPPET_CHARS - 3] + "..."
        else:
            text = hit.ocr_text
        rows.append([hit.rank, hit.image_id, text])
    return rows


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    arm_used: str
    notes: list[str]


class DemoSearch:
    """Ranks the gallery for a query with any of the five arms.

    ``encoder_factory`` builds the text encoder on first use (so the app starts without loading
    CLIP). A query that is exactly a stored caption uses its stored embedding instead.
    """

    def __init__(
        self,
        gallery: Gallery,
        *,
        encoder_factory: Callable[[], Any] | None = None,
        weight: float = DEFAULT_WEIGHT,
        weight_source: str | None = None,
    ) -> None:
        self.gallery = gallery
        self.weight = weight
        self.weight_source = weight_source
        self._factory = encoder_factory
        self._encoder: Any = None
        self._lock = threading.Lock()
        index = MultimodalIndex.from_gallery(gallery, need_bm25=gallery.has_ocr)
        self.retriever = Retriever(index, top_k=100)
        self._stored: dict[str, Query] = {}
        for query in gallery.queries:
            self._stored.setdefault(query.text, query)

    @property
    def has_ocr(self) -> bool:
        return self.gallery.has_ocr

    def weight_note(self) -> str:
        if self.weight_source:
            return (
                f"fuse_score_weighted uses CLIP weight {self.weight:.2f} "
                f"(recorded in {self.weight_source})."
            )
        return (
            f"fuse_score_weighted uses CLIP weight {self.weight:.2f}: no dev results file was "
            "found in results/dev/, so this is the default, not a tuned value."
        )

    def _vector(self, text: str) -> tuple[np.ndarray, str | None, str]:
        stored = self._stored.get(text)
        if stored is not None:
            return self.gallery.query_vector(stored.qid), stored.qid, "stored caption embedding"
        if self._factory is None:
            raise RuntimeError("no text encoder available; pick an example caption instead")
        encoder = self._get_encoder()
        return np.asarray(encoder.encode_query(text), dtype=np.float32), None, "CLIP text"

    def _get_encoder(self) -> Any:
        with self._lock:
            if self._encoder is None:
                self._encoder = self._factory()
            return self._encoder

    def warm_up(self) -> None:
        """Build the text encoder now (run from a background thread at start-up)."""
        if self._factory is not None:
            # a failed warm-up resurfaces, with its real error, on the first query
            with contextlib.suppress(Exception):
                self._get_encoder()

    def search(self, text: str, arm: str, top_k: int = DEFAULT_TOP_K) -> SearchResult:
        text = text.strip()
        if not text:
            raise ValueError("enter a query")
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        notes: list[str] = []
        used = arm
        if arm in OCR_ARMS and not self.has_ocr:
            used = "clip"
            notes.append(f"OCR is not loaded, so {arm} is shown as the CLIP ranking.")
        vector, qid, how = (None, None, "") if used == "ocr_bm25" else self._vector(text)
        if how:
            notes.append(f"Query vector: {how}.")
        if used == "fuse_score_weighted":
            notes.append(self.weight_note())
        ranking = self.retriever.rank(used, text, vector, weight=self.weight, query_key=qid)
        if not ranking:
            notes.append("No image matched (BM25 found no word in common).")
        return SearchResult(format_hits(ranking, self.gallery, top_k), used, notes)
