"""How OCR text becomes a BM25 document, and that the indexes carry image ids."""

from __future__ import annotations

from mmsearch.gallery import OcrLine, OcrRecord
from mmsearch.index import MultimodalIndex, ocr_document


def _record(*lines: tuple[str, float]) -> OcrRecord:
    return OcrRecord("img", tuple(OcrLine(t, c) for t, c in lines))


def test_document_joins_lines_in_order_with_single_spaces():
    rec = _record(("STOP  ", 0.9), ("  main\tstreet", 0.8), ("", 0.99))
    assert ocr_document(rec) == "STOP main street"


def test_no_floor_by_default_keeps_low_confidence_lines():
    rec = _record(("sure", 0.95), ("maybe", 0.05))
    assert ocr_document(rec) == "sure maybe"


def test_confidence_floor_drops_lines_below_it_and_keeps_equal():
    rec = _record(("sure", 0.95), ("maybe", 0.05), ("edge", 0.5))
    assert ocr_document(rec, min_confidence=0.5) == "sure edge"


def test_no_text_gives_empty_document():
    assert ocr_document(_record()) == ""
    assert ocr_document(None) == ""


def test_indexes_return_image_ids(tiny):
    index = MultimodalIndex.from_gallery(tiny)
    textcaps = next(i for i in tiny.items if i.source == "textcaps")
    word = tiny.ocr[textcaps.image_id].lines[0].text
    assert index.ocr_search(word, 5)[0][0] == textcaps.image_id

    vec = tiny.embeddings.image_vectors[0]
    assert index.clip_search(vec, 3)[0][0] == tiny.items[0].image_id


def test_image_without_text_is_never_returned_by_ocr(tiny):
    index = MultimodalIndex.from_gallery(tiny)
    empty = {i for i, r in tiny.ocr.items() if not any(ln.text for ln in r.lines)}
    assert empty
    for query in ("v001", "r0001", "n01", "stop"):
        assert all(image_id not in empty for image_id, _ in index.ocr_search(query, 50))
