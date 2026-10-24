"""Synthetic galleries that let the whole pipeline be tested without any real data.

Two variants share one generator and differ only in what the OCR text contains:

``planted``
    Reading-slice ("textcaps") captions mention words that are printed in the image and so
    appear in its OCR text, but that are *not* part of what the fake image encoder sees. Those
    queries are solvable almost only through OCR; visual-slice ("coco") captions are solvable
    through the image encoder and carry no printed text. A correct pipeline must find fusion
    better than CLIP on the reading slice and indistinguishable on the visual slice.
``null``
    Identical captions and image vectors, but the OCR text is drawn independently of the
    captions, so OCR carries no signal. A correct pipeline must not find a fusion gain.

Words are opaque tokens (``v017`` visible, ``r0421`` printed, ``n07`` shared sign noise), which
keeps the generator free of any real-language ambiguity. Nothing here is a measurement of a real
system: it only checks that the code can detect an effect that was put there on purpose.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mmsearch.embed import FakeEmbedder, FakeImage
from mmsearch.gallery import (
    Embeddings,
    Gallery,
    GalleryItem,
    OcrLine,
    OcrRecord,
    Query,
)

FIXTURE_KINDS = ("planted", "null")

_N_VISUAL, _N_READING, _N_NOISE = 400, 3000, 50


def make_fixture(
    kind: str = "planted",
    *,
    n_per_source: int = 100,
    captions_per_image: int = 5,
    dim: int = 64,
    seed: int = 0,
) -> Gallery:
    """Build an in-memory gallery with embeddings, OCR and a sprinkling of hand tags."""
    if kind not in FIXTURE_KINDS:
        raise ValueError(f"unknown fixture {kind!r}; expected one of {list(FIXTURE_KINDS)}")
    rng = np.random.default_rng(seed)
    items: list[GalleryItem] = []
    queries: list[Query] = []
    ocr: dict[str, OcrRecord] = {}
    shown: list[FakeImage] = []

    for source in ("coco", "textcaps"):
        for i in range(n_per_source):
            image_id = f"{source}-{i:04d}"
            items.append(
                GalleryItem(
                    image_id=image_id,
                    source=source,
                    file_name=f"{image_id}.jpg",
                    url=f"https://example.invalid/{image_id}.jpg",
                    sha256=hashlib.sha256(image_id.encode()).hexdigest(),
                )
            )
            visual = [f"v{w:03d}" for w in rng.choice(_N_VISUAL, 4, replace=False)]
            shown.append(FakeImage(key=image_id, words=tuple(visual)))

            printed: list[str] = []
            if source == "textcaps":
                printed = [f"r{w:04d}" for w in rng.choice(_N_READING, 2, replace=False)]
            ocr_words = printed
            if kind == "null" and source == "textcaps":
                # Same amount of text, unrelated to the captions.
                ocr_words = [f"r{w:04d}" for w in rng.choice(_N_READING, 2, replace=False)]
            lines = [OcrLine(w, round(float(rng.uniform(0.6, 0.99)), 3)) for w in ocr_words]
            if rng.random() < 0.3:
                lines.append(OcrLine(f"n{rng.integers(_N_NOISE):02d}", 0.9))
            ocr[image_id] = OcrRecord(
                image_id, tuple(lines), engine="fixture", preprocessing="none"
            )

            tagged = i % 5 == 0
            for k in range(captions_per_image):
                if source == "coco":
                    words = [str(w) for w in rng.permutation(visual)[:2]]
                    tag = "visual" if tagged else None
                else:
                    n_read = 2 if k % 2 == 0 else 1
                    words = [str(rng.choice(visual))] + [
                        str(w) for w in rng.permutation(printed)[:n_read]
                    ]
                    tag = ("text" if k % 2 == 0 else "both") if tagged else None
                queries.append(
                    Query(
                        qid=f"{image_id}-q{k}",
                        text=" ".join(rng.permutation(words)),
                        image_id=image_id,
                        source=source,
                        tag=tag,
                    )
                )

    embedder = FakeEmbedder(dim)
    embeddings = Embeddings(
        image_vectors=embedder.encode_images(shown),
        query_vectors=embedder.encode_documents([q.text for q in queries]),
        model_name=embedder.name,
        pretrained="fixture",
    )
    return Gallery(items=tuple(items), queries=tuple(queries), ocr=ocr, embeddings=embeddings)


def write_fixture(gallery: Gallery, directory: str | Path, *, shuffle_rows: bool = False) -> Path:
    """Write a gallery in the on-disk contract. ``shuffle_rows`` reverses the embedding rows.

    Reversing checks that loaders align rows by id rather than by position.
    """
    root = Path(directory)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    (root / "ocr").mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    dump(
        root / "gallery.jsonl",
        [
            {
                "image_id": i.image_id,
                "source": i.source,
                "file_name": i.file_name,
                "url": i.url,
                "sha256": i.sha256,
            }
            for i in gallery.items
        ],
    )
    dump(
        root / "queries.jsonl",
        [
            {"qid": q.qid, "text": q.text, "image_id": q.image_id, "source": q.source, "tag": q.tag}
            for q in gallery.queries
        ],
    )
    dump(
        root / "ocr" / "ocr.jsonl",
        [
            {
                "image_id": r.image_id,
                "lines": [{"text": ln.text, "confidence": ln.confidence} for ln in r.lines],
                "engine": r.engine,
                "preprocessing": r.preprocessing,
            }
            for r in gallery.ocr.values()
        ],
    )
    emb = gallery.require_embeddings()
    image_ids, qids = gallery.image_ids, [q.qid for q in gallery.queries]
    img, txt = emb.image_vectors, emb.query_vectors
    if shuffle_rows:
        image_ids, img = image_ids[::-1], img[::-1]
        qids, txt = qids[::-1], txt[::-1]
    np.save(root / "embeddings" / "clip_img.npy", np.ascontiguousarray(img, dtype=np.float32))
    np.save(root / "embeddings" / "clip_txt.npy", np.ascontiguousarray(txt, dtype=np.float32))
    (root / "embeddings" / "meta.json").write_text(
        json.dumps(
            {
                "model_name": emb.model_name,
                "pretrained": emb.pretrained,
                "dim": emb.dim,
                "image_ids": image_ids,
                "qids": qids,
            }
        ),
        encoding="utf-8",
    )
    return root


def write_tags_csv(path: str | Path, tags: dict[str, str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "tag"])
        writer.writerows(sorted(tags.items()))
