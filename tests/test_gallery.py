"""Loader behaviour and every validation failure of the on-disk contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmsearch.fixtures import make_fixture, write_fixture, write_tags_csv
from mmsearch.gallery import DataError, load_gallery


@pytest.fixture
def data_dir(tmp_path: Path, tiny) -> Path:
    return write_fixture(tiny, tmp_path / "data")


def _rewrite_jsonl(path: Path, edit) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    edit(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _rewrite_meta(data_dir: Path, edit) -> None:
    path = data_dir / "embeddings" / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    edit(meta)
    path.write_text(json.dumps(meta), encoding="utf-8")


def test_round_trip_aligns_rows_by_id_not_position(tmp_path, tiny):
    root = write_fixture(tiny, tmp_path / "data", shuffle_rows=True)
    loaded = load_gallery(root)
    assert loaded.image_ids == tiny.image_ids
    assert [q.qid for q in loaded.queries] == [q.qid for q in tiny.queries]
    np.testing.assert_allclose(
        loaded.embeddings.image_vectors, tiny.embeddings.image_vectors, atol=1e-6
    )
    np.testing.assert_allclose(
        loaded.embeddings.query_vectors, tiny.embeddings.query_vectors, atol=1e-6
    )
    assert loaded.ocr.keys() == tiny.ocr.keys()


def test_rows_are_normalised_on_load(data_dir):
    path = data_dir / "embeddings" / "clip_img.npy"
    np.save(path, np.load(path) * 7.5)
    loaded = load_gallery(data_dir)
    norms = np.linalg.norm(loaded.embeddings.image_vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embeddings_and_ocr_are_optional(data_dir):
    for sub in ("embeddings", "ocr"):
        for f in (data_dir / sub).iterdir():
            f.unlink()
    loaded = load_gallery(data_dir)
    assert loaded.embeddings is None
    assert not loaded.has_ocr
    with pytest.raises(DataError, match="no embeddings"):
        loaded.require_embeddings()
    with pytest.raises(DataError, match="no OCR"):
        loaded.require_ocr()


def test_tags_csv_overrides_tag_field(data_dir, tiny):
    qid = tiny.queries[0].qid
    write_tags_csv(data_dir / "tags.csv", {qid: "both"})
    loaded = load_gallery(data_dir)
    assert loaded.queries[0].tag == "both"
    assert loaded.queries[1].tag == tiny.queries[1].tag


def test_blank_tag_in_csv_clears_the_tag(data_dir, tiny):
    tagged = next(q for q in tiny.queries if q.tag)
    (data_dir / "tags.csv").write_text(f"qid,tag\n{tagged.qid},\n", encoding="utf-8")
    loaded = load_gallery(data_dir)
    assert next(q for q in loaded.queries if q.qid == tagged.qid).tag is None


def test_stats(tiny):
    stats = tiny.stats()
    assert stats["images"] == 12
    assert stats["queries"] == 24
    assert stats["per_source"] == {
        "coco": {"images": 6, "queries": 12},
        "textcaps": {"images": 6, "queries": 12},
    }
    assert stats["embedding_dim"] == 16
    assert 0 < stats["images_with_ocr_text"] <= 12


# -- failures ---------------------------------------------------------------
def test_missing_directory(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_gallery(tmp_path / "nope")


def test_missing_gallery_file(data_dir):
    (data_dir / "gallery.jsonl").unlink()
    with pytest.raises(DataError, match="missing file"):
        load_gallery(data_dir)


def test_invalid_json_names_file_and_line(data_dir):
    path = data_dir / "queries.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"queries\.jsonl:25: invalid JSON"):
        load_gallery(data_dir)


def test_duplicate_image_id(data_dir):
    _rewrite_jsonl(data_dir / "gallery.jsonl", lambda rows: rows.append(dict(rows[0])))
    with pytest.raises(DataError, match="duplicate image_id"):
        load_gallery(data_dir)


def test_duplicate_qid(data_dir):
    _rewrite_jsonl(data_dir / "queries.jsonl", lambda rows: rows.append(dict(rows[0])))
    with pytest.raises(DataError, match="duplicate qid"):
        load_gallery(data_dir)


def test_unknown_source(data_dir):
    _rewrite_jsonl(data_dir / "gallery.jsonl", lambda rows: rows[0].update(source="flickr"))
    with pytest.raises(DataError, match="source 'flickr'"):
        load_gallery(data_dir)


def test_missing_required_key(data_dir):
    _rewrite_jsonl(data_dir / "queries.jsonl", lambda rows: rows[0].pop("text"))
    with pytest.raises(DataError, match="missing required key 'text'"):
        load_gallery(data_dir)


def test_invalid_tag(data_dir):
    _rewrite_jsonl(data_dir / "queries.jsonl", lambda rows: rows[0].update(tag="maybe"))
    with pytest.raises(DataError, match="tag 'maybe'"):
        load_gallery(data_dir)


def test_query_points_at_unknown_image(data_dir):
    _rewrite_jsonl(data_dir / "queries.jsonl", lambda rows: rows[0].update(image_id="ghost"))
    with pytest.raises(DataError, match="not in the gallery"):
        load_gallery(data_dir)


def test_query_source_disagrees_with_image(data_dir):
    _rewrite_jsonl(data_dir / "queries.jsonl", lambda rows: rows[0].update(source="textcaps"))
    with pytest.raises(DataError, match="differs from its image"):
        load_gallery(data_dir)


def test_tags_csv_with_unknown_qid(data_dir):
    (data_dir / "tags.csv").write_text("qid,tag\nghost,visual\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"not in queries.jsonl"):
        load_gallery(data_dir)


def test_tags_csv_bad_header_and_duplicates(data_dir, tiny):
    (data_dir / "tags.csv").write_text("id,label\nx,visual\n", encoding="utf-8")
    with pytest.raises(DataError, match="header"):
        load_gallery(data_dir)
    qid = tiny.queries[0].qid
    (data_dir / "tags.csv").write_text(f"qid,tag\n{qid},visual\n{qid},text\n", encoding="utf-8")
    with pytest.raises(DataError, match="duplicate qid"):
        load_gallery(data_dir)


def test_meta_image_ids_mismatch(data_dir):
    _rewrite_meta(data_dir, lambda m: m["image_ids"].__setitem__(0, "ghost"))
    with pytest.raises(DataError, match="image_ids do not match"):
        load_gallery(data_dir)


def test_meta_qids_mismatch(data_dir):
    _rewrite_meta(data_dir, lambda m: m["qids"].pop())
    with pytest.raises(DataError, match="qids do not match"):
        load_gallery(data_dir)


def test_meta_duplicate_ids(data_dir):
    _rewrite_meta(data_dir, lambda m: m["image_ids"].append(m["image_ids"][0]))
    with pytest.raises(DataError, match="duplicate entries in image_ids"):
        load_gallery(data_dir)


def test_shape_mismatch_with_meta_dim(data_dir):
    _rewrite_meta(data_dir, lambda m: m.update(dim=m["dim"] + 1))
    with pytest.raises(DataError, match="shape"):
        load_gallery(data_dir)


def test_non_finite_embeddings(data_dir):
    path = data_dir / "embeddings" / "clip_txt.npy"
    arr = np.load(path)
    arr[0, 0] = np.nan
    np.save(path, arr)
    with pytest.raises(DataError, match="NaN"):
        load_gallery(data_dir)


def test_incomplete_embeddings(data_dir):
    (data_dir / "embeddings" / "meta.json").unlink()
    with pytest.raises(DataError, match="incomplete embeddings"):
        load_gallery(data_dir)


def test_ocr_must_cover_every_image(data_dir):
    _rewrite_jsonl(data_dir / "ocr" / "ocr.jsonl", lambda rows: rows.pop())
    with pytest.raises(DataError, match=r"covers 11 of 12 images"):
        load_gallery(data_dir)


def test_ocr_duplicate_image(data_dir):
    _rewrite_jsonl(data_dir / "ocr" / "ocr.jsonl", lambda rows: rows.append(dict(rows[0])))
    with pytest.raises(DataError, match="duplicate image_id"):
        load_gallery(data_dir)


def test_ocr_unknown_image(tmp_path):
    root = write_fixture(
        make_fixture("planted", n_per_source=6, captions_per_image=2, dim=16), tmp_path
    )
    _rewrite_jsonl(root / "ocr" / "ocr.jsonl", lambda rows: rows[0].update(image_id="ghost"))
    with pytest.raises(DataError, match="not in the gallery"):
        load_gallery(root)


def test_ocr_line_shape(data_dir):
    _rewrite_jsonl(
        data_dir / "ocr" / "ocr.jsonl", lambda rows: rows[0].update(lines=[{"text": "x"}])
    )
    with pytest.raises(DataError, match="numeric 'confidence'"):
        load_gallery(data_dir)
