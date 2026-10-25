"""The tag-sheet converter: blind ids mapped back to qids, and every bad sheet refused."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "import_tags.py"
_spec = importlib.util.spec_from_file_location("import_tags", SCRIPT)
import_tags = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(import_tags)


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def data(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    _write(
        root / "tagging" / "tag_sheet.csv",
        ["qid", "text", "image_path", "tag"],
        [["t001", "a dog", "x.jpg", "visual"], ["t002", "a sign", "y.jpg", " Text "]],
    )
    _write(
        root / "tagging" / "tag_key.csv",
        ["blind_id", "qid"],
        [["t001", "coco-1"], ["t002", "tc-abc-0"]],
    )
    (root / "queries.jsonl").write_text(
        "".join(json.dumps({"qid": q}) + "\n" for q in ("coco-1", "tc-abc-0", "coco-2")),
        encoding="utf-8",
    )
    return root


def _set_tag(root: Path, blind: str, tag: str) -> None:
    path = root / "tagging" / "tag_sheet.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        if row["qid"] == blind:
            row["tag"] = tag
    _write(path, list(rows[0]), [list(r.values()) for r in rows])


def test_writes_real_qids_with_normalised_labels(data: Path, capsys) -> None:
    assert import_tags.main(["--data-dir", str(data)]) == 0
    rows = list(csv.DictReader((data / "tags.csv").open(encoding="utf-8")))
    assert rows == [{"qid": "coco-1", "tag": "visual"}, {"qid": "tc-abc-0", "tag": "text"}]
    assert "2 tags" in capsys.readouterr().out


def test_check_writes_nothing(data: Path) -> None:
    assert import_tags.main(["--data-dir", str(data), "--check"]) == 0
    assert not (data / "tags.csv").exists()


def test_incomplete_sheet_is_refused_unless_partial(data: Path) -> None:
    _set_tag(data, "t002", "")
    assert import_tags.main(["--data-dir", str(data)]) == 2
    assert not (data / "tags.csv").exists()
    assert import_tags.main(["--data-dir", str(data), "--allow-partial"]) == 0
    assert (data / "tags.csv").read_text(encoding="utf-8").count("\n") == 2


def test_unknown_label_is_refused(data: Path) -> None:
    _set_tag(data, "t001", "scene")
    with pytest.raises(import_tags.TagError, match="scene"):
        import_tags.convert(
            data / "tagging" / "tag_sheet.csv",
            data / "tagging" / "tag_key.csv",
            data / "queries.jsonl",
        )


def test_sheet_and_key_must_agree(data: Path) -> None:
    _write(data / "tagging" / "tag_key.csv", ["blind_id", "qid"], [["t001", "coco-1"]])
    assert import_tags.main(["--data-dir", str(data)]) == 2


def test_qid_missing_from_queries_is_refused(data: Path) -> None:
    _write(
        data / "tagging" / "tag_key.csv",
        ["blind_id", "qid"],
        [["t001", "coco-1"], ["t002", "tc-zzz-9"]],
    )
    assert import_tags.main(["--data-dir", str(data)]) == 2


def test_existing_tags_file_needs_force(data: Path) -> None:
    assert import_tags.main(["--data-dir", str(data)]) == 0
    assert import_tags.main(["--data-dir", str(data)]) == 2
    assert import_tags.main(["--data-dir", str(data), "--force"]) == 0


def test_empty_sheet_is_refused(data: Path) -> None:
    _set_tag(data, "t001", "")
    _set_tag(data, "t002", "")
    assert import_tags.main(["--data-dir", str(data), "--allow-partial"]) == 2
