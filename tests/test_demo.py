"""Demo helpers: result formatting, loading with and without OCR, and the search wrapper.

Only torch-free and gradio-free code runs unconditionally; anything needing gradio is skipped
when the ``demo`` extra is absent. No real data and no model are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `demo` lives at the repository root, outside src/
    sys.path.insert(0, str(ROOT))

from demo.core import (  # noqa: E402
    DemoSearch,
    example_choices,
    format_hits,
    gallery_items,
    load_demo_gallery,
    ocr_rows,
    recorded_weight,
)

from mmsearch.embed import FakeEmbedder  # noqa: E402
from mmsearch.fixtures import make_fixture  # noqa: E402
from mmsearch.gallery import DataError, Gallery  # noqa: E402


@pytest.fixture
def gallery() -> Gallery:
    return make_fixture("planted", n_per_source=6, captions_per_image=2, dim=16)


def test_format_hits_carries_url_source_score_and_ocr(gallery: Gallery) -> None:
    image_ids = gallery.image_ids[:3]
    hits = format_hits([(i, 0.5 - 0.1 * n) for n, i in enumerate(image_ids)], gallery, top_k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].image_id == image_ids[0]
    assert hits[0].url == gallery.item(image_ids[0]).url
    assert hits[0].source == gallery.item(image_ids[0]).source
    assert hits[0].caption == f"#1 | 0.500 | {hits[0].source} | {image_ids[0]}"
    assert gallery_items(hits) == [(h.url, h.caption) for h in hits]


def test_hits_without_ocr_are_marked_not_loaded(gallery: Gallery) -> None:
    gallery.ocr = {}
    hits = format_hits([(gallery.image_ids[0], 1.0)], gallery)
    assert hits[0].ocr_text is None
    assert ocr_rows(hits)[0][2] == "(OCR not loaded)"


def test_ocr_rows_empty_and_long_text(gallery: Gallery) -> None:
    from mmsearch.gallery import OcrLine, OcrRecord

    a, b = gallery.image_ids[:2]
    gallery.ocr = {i: OcrRecord(i) for i in gallery.image_ids}
    gallery.ocr[b] = OcrRecord(b, (OcrLine("word " * 100, 0.9),))
    rows = ocr_rows(format_hits([(a, 1.0), (b, 0.5)], gallery))
    assert rows[0] == [1, a, "(no text found)"]
    assert rows[1][2].endswith("...")
    assert len(rows[1][2]) <= 160


def test_top_k_larger_than_ranking_and_zero(gallery: Gallery) -> None:
    ranking = [(gallery.image_ids[0], 1.0)]
    assert len(format_hits(ranking, gallery, top_k=50)) == 1
    assert format_hits(ranking, gallery, top_k=0) == []


def test_example_choices_take_a_few_from_each_source(gallery: Gallery) -> None:
    choices = example_choices(gallery.queries, per_source=3)
    sources = [label.split("]")[0].lstrip("[") for label, _ in choices]
    assert sources.count("coco") == 3
    assert sources.count("textcaps") == 3
    qids = {q.qid for q in gallery.queries}
    assert all(qid in qids for _, qid in choices)


def test_recorded_weight_reads_dev_results_only(tmp_path: Path) -> None:
    assert recorded_weight(tmp_path / "missing") == (None, None)
    dev = tmp_path / "dev"
    dev.mkdir()
    assert recorded_weight(dev) == (None, None)
    (dev / "bad.json").write_text("{not json", encoding="utf-8")
    (dev / "eval.json").write_text(
        json.dumps({"config": {"weight": {"chosen_clip_weight": 0.35}}}), encoding="utf-8"
    )
    assert recorded_weight(dev) == (0.35, "eval.json")
    (dev / "eval.json").write_text(
        json.dumps({"config": {"weight": {"chosen_clip_weight": 7}}}), encoding="utf-8"
    )
    assert recorded_weight(dev) == (None, None)


def _write_data_dir(root: Path, gallery: Gallery, *, ocr: str) -> None:
    """Write ``gallery`` as a data directory. ``ocr`` is 'full', 'partial' or 'none'."""
    (root / "embeddings").mkdir(parents=True)
    emb = gallery.require_embeddings()
    np.save(root / "embeddings" / "clip_img.npy", emb.image_vectors)
    np.save(root / "embeddings" / "clip_txt.npy", emb.query_vectors)
    meta = {
        "model_name": "fake",
        "dim": emb.dim,
        "image_ids": gallery.image_ids,
        "qids": [q.qid for q in gallery.queries],
    }
    (root / "embeddings" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    items = [
        {"image_id": i.image_id, "source": i.source, "file_name": i.file_name, "url": i.url}
        for i in gallery.items
    ]
    queries = [
        {"qid": q.qid, "text": q.text, "image_id": q.image_id, "source": q.source, "tag": None}
        for q in gallery.queries
    ]
    (root / "gallery.jsonl").write_text("\n".join(map(json.dumps, items)) + "\n", "utf-8")
    (root / "queries.jsonl").write_text("\n".join(map(json.dumps, queries)) + "\n", "utf-8")
    if ocr != "none":
        (root / "ocr").mkdir()
        ids = gallery.image_ids if ocr == "full" else gallery.image_ids[:3]
        rows = [{"image_id": i, "lines": [{"text": f"text {i}", "confidence": 0.9}]} for i in ids]
        (root / "ocr" / "ocr.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n", "utf-8")


def test_load_without_ocr_file_runs_clip_only_with_notice(tmp_path: Path, gallery: Gallery) -> None:
    _write_data_dir(tmp_path, gallery, ocr="none")
    loaded, notices = load_demo_gallery(tmp_path)
    assert not loaded.has_ocr
    assert len(notices) == 1
    assert "CLIP-only" in notices[0]


def test_load_with_partial_ocr_drops_it_with_notice(tmp_path: Path, gallery: Gallery) -> None:
    _write_data_dir(tmp_path, gallery, ocr="partial")
    loaded, notices = load_demo_gallery(tmp_path)
    assert not loaded.has_ocr
    assert "3 of" in notices[0]


def test_load_with_full_ocr_has_no_notice(tmp_path: Path, gallery: Gallery) -> None:
    _write_data_dir(tmp_path, gallery, ocr="full")
    loaded, notices = load_demo_gallery(tmp_path)
    assert loaded.has_ocr
    assert notices == []


def test_load_needs_embeddings(tmp_path: Path, gallery: Gallery) -> None:
    _write_data_dir(tmp_path, gallery, ocr="none")
    for name in ("clip_img.npy", "clip_txt.npy", "meta.json"):
        (tmp_path / "embeddings" / name).unlink()
    with pytest.raises(DataError, match="embeddings"):
        load_demo_gallery(tmp_path)


def test_search_uses_stored_vector_for_a_stored_caption(gallery: Gallery) -> None:
    calls: list[str] = []

    def factory():
        calls.append("built")
        return FakeEmbedder(gallery.require_embeddings().dim)

    engine = DemoSearch(gallery, encoder_factory=factory)
    query = gallery.queries[0]
    result = engine.search(query.text, "clip", top_k=5)
    assert calls == []  # the encoder was never needed
    assert "stored caption embedding" in " ".join(result.notes)
    assert len(result.hits) == 5
    assert result.hits[0].score >= result.hits[-1].score


def test_search_free_text_builds_the_encoder_once(gallery: Gallery) -> None:
    built: list[int] = []

    def factory():
        built.append(1)
        return FakeEmbedder(gallery.require_embeddings().dim)

    engine = DemoSearch(gallery, encoder_factory=factory)
    engine.search("something not stored", "clip")
    engine.search("another new query", "fuse_rrf")
    assert built == [1]


def test_search_without_encoder_or_stored_caption_explains_itself(gallery: Gallery) -> None:
    engine = DemoSearch(gallery, encoder_factory=None)
    with pytest.raises(RuntimeError, match="example caption"):
        engine.search("not a stored caption", "clip")
    with pytest.raises(ValueError, match="enter a query"):
        engine.search("   ", "clip")


@pytest.mark.parametrize("arm", ["ocr_bm25", "fuse_rrf", "fuse_score", "fuse_score_weighted"])
def test_ocr_arms_fall_back_to_clip_when_ocr_is_missing(gallery: Gallery, arm: str) -> None:
    gallery.ocr = {}
    engine = DemoSearch(gallery, encoder_factory=None)
    text = gallery.queries[0].text
    result = engine.search(text, arm)
    assert result.arm_used == "clip"
    assert any("OCR is not loaded" in n for n in result.notes)
    assert result.hits == engine.search(text, "clip").hits


def test_all_arms_run_with_ocr(gallery: Gallery) -> None:
    engine = DemoSearch(gallery, encoder_factory=None)
    text = gallery.queries[0].text
    for arm in ("clip", "ocr_bm25", "fuse_rrf", "fuse_score", "fuse_score_weighted"):
        result = engine.search(text, arm, top_k=4)
        assert result.arm_used == arm
        assert len(result.hits) <= 4
        assert [h.rank for h in result.hits] == list(range(1, len(result.hits) + 1))


def test_weighted_arm_note_says_default_or_recorded(gallery: Gallery) -> None:
    default = DemoSearch(gallery)
    assert "default, not a tuned value" in default.weight_note()
    tuned = DemoSearch(gallery, weight=0.3, weight_source="results/dev/eval.json")
    assert "0.30" in tuned.weight_note()
    assert "results/dev/eval.json" in tuned.weight_note()
    text = gallery.queries[0].text
    notes = tuned.search(text, "fuse_score_weighted").notes
    assert any("0.30" in n for n in notes)


def test_encoder_module_imports_without_torch() -> None:
    import demo.encoder as encoder

    assert callable(encoder.factory)
    assert encoder.CONTEXT_LENGTH == 77


def test_app_module_imports_without_gradio() -> None:
    import demo.app as app

    assert callable(app.build_app)


def test_app_builds_with_gradio(tmp_path: Path, gallery: Gallery) -> None:
    pytest.importorskip("gradio")
    _write_data_dir(tmp_path, gallery, ocr="none")
    from demo.app import build_app

    app = build_app(tmp_path, tmp_path / "results" / "dev", warm_up=False)
    assert app is not None
