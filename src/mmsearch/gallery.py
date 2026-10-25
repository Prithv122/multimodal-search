"""The on-disk data contract and its loaders.

Everything the study needs lives under one data directory (default ``data/``):

``gallery.jsonl``
    One image per line: ``{"image_id", "source", "file_name", "url", "sha256"}``.
    ``source`` is ``"coco"`` or ``"textcaps"``. Extra keys are ignored.
``queries.jsonl``
    One query per line: ``{"qid", "text", "image_id", "source", "tag"}``. Every human
    caption is a query and its image is the single relevant item. ``tag`` is ``null`` or
    one of ``visual`` / ``text`` / ``both``.
``tags.csv`` (optional)
    ``qid,tag`` owner hand tags. When present it overrides the ``tag`` field.
``embeddings/clip_img.npy``, ``embeddings/clip_txt.npy``, ``embeddings/meta.json`` (optional)
    float32 matrices whose rows follow ``meta["image_ids"]`` and ``meta["qids"]``. Rows
    need not be unit length; they are normalised on load and re-ordered to the gallery order.
``ocr/ocr.jsonl`` (optional)
    ``{"image_id", "lines": [{"text", "confidence"}], "engine", "preprocessing"}``. An image
    with no text has ``lines: []``. When the file exists it must cover every gallery image.

Validation is strict and loud: every problem raises :class:`DataError` naming the file (and
line, where there is one), because a silently wrong gallery produces a plausible-looking and
wrong result table.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mmsearch._vendor.dense import l2_normalise

SOURCES = ("coco", "textcaps")
TAGS = ("visual", "text", "both")

GALLERY_FILE = "gallery.jsonl"
QUERIES_FILE = "queries.jsonl"
TAGS_FILE = "tags.csv"
IMG_VECTORS_FILE = "embeddings/clip_img.npy"
TXT_VECTORS_FILE = "embeddings/clip_txt.npy"
META_FILE = "embeddings/meta.json"
OCR_FILE = "ocr/ocr.jsonl"


class DataError(ValueError):
    """The data directory does not satisfy the contract."""


@dataclass(frozen=True)
class GalleryItem:
    image_id: str
    source: str
    file_name: str = ""
    url: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    image_id: str
    source: str
    tag: str | None = None


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float


@dataclass(frozen=True)
class OcrRecord:
    image_id: str
    lines: tuple[OcrLine, ...] = ()
    engine: str = ""
    preprocessing: str = ""


@dataclass(frozen=True)
class Embeddings:
    """Unit-length vectors aligned to ``Gallery.items`` and ``Gallery.queries``."""

    image_vectors: np.ndarray
    query_vectors: np.ndarray
    model_name: str
    pretrained: str = ""

    @property
    def dim(self) -> int:
        return int(self.image_vectors.shape[1])


@dataclass
class Gallery:
    items: tuple[GalleryItem, ...]
    queries: tuple[Query, ...]
    ocr: dict[str, OcrRecord] = field(default_factory=dict)
    embeddings: Embeddings | None = None
    _item_index: dict[str, int] = field(default_factory=dict, repr=False)
    _query_index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._item_index = {item.image_id: i for i, item in enumerate(self.items)}
        self._query_index = {q.qid: j for j, q in enumerate(self.queries)}

    @property
    def image_ids(self) -> list[str]:
        return [item.image_id for item in self.items]

    @property
    def has_ocr(self) -> bool:
        return bool(self.ocr)

    def item(self, image_id: str) -> GalleryItem:
        return self.items[self._item_index[image_id]]

    def query_vector(self, qid: str) -> np.ndarray:
        if self.embeddings is None:
            raise DataError("this gallery has no embeddings loaded")
        return self.embeddings.query_vectors[self._query_index[qid]]

    def require_embeddings(self) -> Embeddings:
        if self.embeddings is None:
            raise DataError(
                f"no embeddings found: expected {IMG_VECTORS_FILE}, {TXT_VECTORS_FILE} and "
                f"{META_FILE} under the data directory"
            )
        return self.embeddings

    def require_ocr(self) -> dict[str, OcrRecord]:
        if not self.ocr:
            raise DataError(f"no OCR found: expected {OCR_FILE} under the data directory")
        return self.ocr

    def stats(self) -> dict[str, Any]:
        """Counts for the ``stats`` command."""
        per_source = {
            s: {
                "images": sum(1 for i in self.items if i.source == s),
                "queries": sum(1 for q in self.queries if q.source == s),
            }
            for s in SOURCES
        }
        tags = Counter(q.tag for q in self.queries if q.tag is not None)
        return {
            "images": len(self.items),
            "queries": len(self.queries),
            "per_source": per_source,
            "tagged_queries": {t: tags.get(t, 0) for t in TAGS},
            "ocr_records": len(self.ocr),
            "images_with_ocr_text": sum(
                1 for rec in self.ocr.values() if any(line.text.strip() for line in rec.lines)
            ),
            "embedding_dim": self.embeddings.dim if self.embeddings else None,
            "embedding_model": self.embeddings.model_name if self.embeddings else None,
        }


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise DataError(f"missing file: {path}")
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise DataError(f"{path}:{number}: expected a JSON object per line")
            yield number, row


def _text_field(row: Mapping[str, Any], key: str, path: Path, number: int) -> str:
    if key not in row:
        raise DataError(f"{path}:{number}: missing required key {key!r}")
    value = row[key]
    if not isinstance(value, str):
        raise DataError(f"{path}:{number}: {key!r} must be a string, got {type(value).__name__}")
    return value


def _check_source(value: str, path: Path, number: int) -> str:
    if value not in SOURCES:
        raise DataError(f"{path}:{number}: source {value!r} is not one of {list(SOURCES)}")
    return value


def _check_tag(value: Any, path: Path, number: int) -> str | None:
    if value is None or value == "":
        return None
    if value not in TAGS:
        raise DataError(f"{path}:{number}: tag {value!r} is not null or one of {list(TAGS)}")
    return str(value)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_items(path: Path) -> tuple[GalleryItem, ...]:
    items: list[GalleryItem] = []
    seen: set[str] = set()
    for number, row in _read_jsonl(path):
        image_id = _text_field(row, "image_id", path, number)
        if image_id in seen:
            raise DataError(f"{path}:{number}: duplicate image_id {image_id!r}")
        seen.add(image_id)
        source = _check_source(_text_field(row, "source", path, number), path, number)
        items.append(
            GalleryItem(
                image_id=image_id,
                source=source,
                file_name=str(row.get("file_name", "")),
                url=str(row.get("url", "")),
                sha256=str(row.get("sha256", "")),
            )
        )
    if not items:
        raise DataError(f"{path}: the gallery is empty")
    return tuple(items)


def load_queries(path: Path) -> tuple[Query, ...]:
    queries: list[Query] = []
    seen: set[str] = set()
    for number, row in _read_jsonl(path):
        qid = _text_field(row, "qid", path, number)
        if qid in seen:
            raise DataError(f"{path}:{number}: duplicate qid {qid!r}")
        seen.add(qid)
        queries.append(
            Query(
                qid=qid,
                text=_text_field(row, "text", path, number),
                image_id=_text_field(row, "image_id", path, number),
                source=_check_source(_text_field(row, "source", path, number), path, number),
                tag=_check_tag(row.get("tag"), path, number),
            )
        )
    if not queries:
        raise DataError(f"{path}: there are no queries")
    return tuple(queries)


def load_tags(path: Path) -> dict[str, str]:
    """Read ``tags.csv`` (``qid,tag``). A blank tag is an explicit "untagged"."""
    tags: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {"qid", "tag"} - set(reader.fieldnames):
            raise DataError(f"{path}: header must contain the columns qid,tag")
        for number, row in enumerate(reader, start=2):
            qid = (row["qid"] or "").strip()
            if not qid:
                raise DataError(f"{path}:{number}: empty qid")
            if qid in tags:
                raise DataError(f"{path}:{number}: duplicate qid {qid!r}")
            tags[qid] = _check_tag((row["tag"] or "").strip(), path, number) or ""
    return tags


def load_ocr(path: Path) -> dict[str, OcrRecord]:
    records: dict[str, OcrRecord] = {}
    for number, row in _read_jsonl(path):
        image_id = _text_field(row, "image_id", path, number)
        if image_id in records:
            raise DataError(f"{path}:{number}: duplicate image_id {image_id!r}")
        raw_lines = row.get("lines")
        if not isinstance(raw_lines, list):
            raise DataError(f"{path}:{number}: 'lines' must be a list")
        lines: list[OcrLine] = []
        for line in raw_lines:
            if not isinstance(line, dict) or not isinstance(line.get("text"), str):
                raise DataError(f"{path}:{number}: each line needs a string 'text'")
            confidence = line.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, int | float):
                raise DataError(f"{path}:{number}: each line needs a numeric 'confidence'")
            lines.append(OcrLine(text=line["text"], confidence=float(confidence)))
        records[image_id] = OcrRecord(
            image_id=image_id,
            lines=tuple(lines),
            engine=str(row.get("engine", "")),
            preprocessing=str(row.get("preprocessing", "")),
        )
    return records


def _check_unique(ids: Sequence[Any], what: str, path: Path) -> None:
    if not all(isinstance(i, str) for i in ids):
        raise DataError(f"{path}: every entry of {what} must be a string")
    if len(set(ids)) != len(ids):
        dupes = [k for k, n in Counter(ids).items() if n > 1][:3]
        raise DataError(f"{path}: duplicate entries in {what}, e.g. {dupes}")


def _align_rows(matrix: np.ndarray, row_ids: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    index = {row_id: i for i, row_id in enumerate(row_ids)}
    return matrix[[index[w] for w in wanted]]


def load_embeddings(root: Path, image_ids: Sequence[str], qids: Sequence[str]) -> Embeddings | None:
    """Load and validate the CLIP arrays; ``None`` when none of the three files exist."""
    img_path, txt_path, meta_path = (
        root / IMG_VECTORS_FILE,
        root / TXT_VECTORS_FILE,
        root / META_FILE,
    )
    paths = (img_path, txt_path, meta_path)
    present = [p.is_file() for p in paths]
    if not any(present):
        return None
    if not all(present):
        missing = [p.name for p, ok in zip(paths, present, strict=True) if not ok]
        raise DataError(f"{root / 'embeddings'}: incomplete embeddings, missing {missing}")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{meta_path}: invalid JSON ({exc.msg})") from exc
    for key in ("model_name", "dim", "image_ids", "qids"):
        if key not in meta:
            raise DataError(f"{meta_path}: missing required key {key!r}")
    meta_images, meta_qids = list(meta["image_ids"]), list(meta["qids"])
    _check_unique(meta_images, "image_ids", meta_path)
    _check_unique(meta_qids, "qids", meta_path)

    for what, got, want in (("image_ids", meta_images, image_ids), ("qids", meta_qids, qids)):
        if set(got) != set(want):
            only_meta = sorted(set(got) - set(want))[:3]
            only_data = sorted(set(want) - set(got))[:3]
            raise DataError(
                f"{meta_path}: {what} do not match the data files "
                f"(only in meta: {only_meta}; only in data: {only_data})"
            )

    img = np.load(img_path, allow_pickle=False)
    txt = np.load(txt_path, allow_pickle=False)
    dim = int(meta["dim"])
    if img.ndim != 2 or img.shape != (len(meta_images), dim):
        raise DataError(f"{img_path}: shape {img.shape} != ({len(meta_images)}, {dim}) from meta")
    if txt.ndim != 2 or txt.shape != (len(meta_qids), dim):
        raise DataError(f"{txt_path}: shape {txt.shape} != ({len(meta_qids)}, {dim}) from meta")
    for path, array in ((img_path, img), (txt_path, txt)):
        if not np.isfinite(array).all():
            raise DataError(f"{path}: contains NaN or infinite values")

    img = _align_rows(img.astype(np.float32), meta_images, image_ids)
    txt = _align_rows(txt.astype(np.float32), meta_qids, qids)
    return Embeddings(
        image_vectors=l2_normalise(img).astype(np.float32),
        query_vectors=l2_normalise(txt).astype(np.float32),
        model_name=str(meta["model_name"]),
        pretrained=str(meta.get("pretrained", "")),
    )


def validate_gallery(gallery: Gallery) -> None:
    """Cross-file checks that need the loaded objects."""
    image_ids = set(gallery.image_ids)
    for query in gallery.queries:
        if query.image_id not in image_ids:
            raise DataError(
                f"query {query.qid!r}: image_id {query.image_id!r} is not in the gallery"
            )
        image_source = gallery.item(query.image_id).source
        if image_source != query.source:
            raise DataError(
                f"query {query.qid!r}: source {query.source!r} differs from its image's "
                f"source {image_source!r}"
            )
    if gallery.ocr:
        missing = sorted(image_ids - set(gallery.ocr))
        extra = sorted(set(gallery.ocr) - image_ids)
        if extra:
            raise DataError(f"{OCR_FILE}: image_ids not in the gallery, e.g. {extra[:3]}")
        if missing:
            raise DataError(
                f"{OCR_FILE}: covers {len(gallery.ocr)} of {len(image_ids)} images; images "
                f"without text must be present with lines [], e.g. missing {missing[:3]}"
            )
    emb = gallery.embeddings
    if emb is not None and (
        emb.image_vectors.shape[0] != len(gallery.items)
        or emb.query_vectors.shape[0] != len(gallery.queries)
        or emb.image_vectors.shape[1] != emb.query_vectors.shape[1]
    ):
        raise DataError("embeddings: shapes disagree with the gallery or between image and text")


def load_gallery(data_dir: str | Path = "data", *, use_tags: bool = True) -> Gallery:
    """Read and validate a data directory. Embeddings, OCR and tags are optional on disk.

    ``use_tags=False`` ignores ``tags.csv``, so a run saved before the tags existed can be
    recomputed exactly as it was scored.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise DataError(f"data directory not found: {root}")
    items = load_items(root / GALLERY_FILE)
    queries = load_queries(root / QUERIES_FILE)

    tags_path = root / TAGS_FILE
    if use_tags and tags_path.is_file():
        tags = load_tags(tags_path)
        unknown = sorted(set(tags) - {q.qid for q in queries})
        if unknown:
            raise DataError(f"{tags_path}: qids not in {QUERIES_FILE}, e.g. {unknown[:3]}")
        queries = tuple(
            Query(q.qid, q.text, q.image_id, q.source, (tags[q.qid] or None))
            if q.qid in tags
            else q
            for q in queries
        )

    ocr_path = root / OCR_FILE
    ocr = load_ocr(ocr_path) if ocr_path.is_file() else {}
    embeddings = load_embeddings(root, [i.image_id for i in items], [q.qid for q in queries])
    gallery = Gallery(items=items, queries=queries, ocr=ocr, embeddings=embeddings)
    validate_gallery(gallery)
    return gallery
