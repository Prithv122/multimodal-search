"""The command line, end to end on the fixtures and on a data directory."""

from __future__ import annotations

import json

import pytest

from mmsearch import cli
from mmsearch.cli import main
from mmsearch.fixtures import make_fixture, write_fixture

_real_make_fixture = cli.make_fixture


@pytest.fixture(autouse=True)
def small_fixtures(monkeypatch):
    """Keep CLI runs quick: the fixture flag builds a 80-image gallery here."""
    monkeypatch.setattr(
        cli,
        "make_fixture",
        lambda kind: _real_make_fixture(kind, n_per_source=40, captions_per_image=3),
    )


def test_stats_on_fixture(capsys):
    assert main(["stats", "--fixture", "planted"]) == 0
    out = capsys.readouterr().out
    assert "images:  80" in out
    assert "queries: 240" in out
    assert "coco" in out and "textcaps" in out
    assert "dim=64" in out
    assert "dev 16 images / test 64" in out


def test_stats_on_data_dir(tmp_path, capsys):
    root = write_fixture(
        make_fixture("null", n_per_source=5, captions_per_image=2, dim=8), tmp_path
    )
    assert main(["stats", "--data-dir", str(root)]) == 0
    assert "images:  10" in capsys.readouterr().out


def test_eval_prints_table_and_writes_json(tmp_path, capsys):
    out_path = tmp_path / "nested" / "results.json"
    code = main(
        [
            "eval",
            "--fixture",
            "planted",
            "--split",
            "test",
            "--resamples",
            "200",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "H1 reading: PASS" in text
    assert "H2 visual : PASS" in text
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(payload) >= {"config", "rows"}
    assert payload["config"]["resamples"] == 200
    row = next(r for r in payload["rows"] if r["kind"] == "arm")
    assert set(row) == {
        "kind",
        "arm",
        "slice",
        "metric",
        "mean",
        "ci_lo",
        "ci_hi",
        "n_images",
        "n_queries",
    }


def test_eval_null_reports_fail_but_exits_zero_unless_asked(capsys):
    args = ["eval", "--fixture", "null", "--resamples", "200"]
    assert main(args) == 0
    assert "H1 reading: FAIL" in capsys.readouterr().out
    assert main([*args, "--fail-on-prereg"]) == 1


def test_eval_non_test_split_is_flagged(capsys):
    main(["eval", "--fixture", "planted", "--split", "dev", "--resamples", "100"])
    assert "not confirmatory" in capsys.readouterr().out.lower()


def test_eval_arm_filter(capsys):
    main(["eval", "--fixture", "planted", "--resamples", "100", "--arm", "clip", "fuse_rrf"])
    out = capsys.readouterr().out
    assert "fuse_rrf" in out
    assert "ocr_bm25  " not in out


def test_search_with_stored_caption_vector(capsys):
    gallery = cli.make_fixture("planted")
    query = next(q for q in gallery.queries if q.source == "textcaps")
    code = main(
        [
            "search",
            query.text,
            "--fixture",
            "planted",
            "--arm",
            "fuse_rrf",
            "--query-qid",
            query.qid,
        ]
    )
    assert code == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert query.image_id in first


def test_search_ocr_arm_needs_no_vector(capsys):
    gallery = cli.make_fixture("planted")
    query = next(q for q in gallery.queries if q.source == "textcaps")
    assert main(["search", query.text, "--fixture", "planted", "--arm", "ocr_bm25"]) == 0
    assert query.image_id in capsys.readouterr().out.splitlines()[0]


def test_search_with_pluggable_encoder(capsys):
    assert (
        main(["search", "v001 v002", "--fixture", "planted", "--arm", "clip", "--top-k", "3"]) == 0
    )
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
    code = main(
        [
            "search",
            "v001",
            "--fixture",
            "planted",
            "--arm",
            "clip",
            "--encoder",
            "mmsearch.embed:FakeEmbedder",
            "--top-k",
            "2",
        ]
    )
    assert code == 0


def test_search_on_data_dir_requires_a_vector_source(tmp_path, capsys):
    root = write_fixture(
        make_fixture("planted", n_per_source=5, captions_per_image=2, dim=8), tmp_path
    )
    assert main(["search", "v001", "--data-dir", str(root), "--arm", "clip"]) == 2
    assert "--query-qid or --encoder" in capsys.readouterr().err
    assert (
        main(["search", "v001", "--data-dir", str(root), "--arm", "clip", "--query-qid", "ghost"])
        == 2
    )


def test_missing_data_dir_is_a_clear_error(tmp_path, capsys):
    assert main(["stats", "--data-dir", str(tmp_path / "absent")]) == 2
    assert "error: data directory not found" in capsys.readouterr().err


def test_fixture_and_data_dir_are_exclusive():
    with pytest.raises(SystemExit):
        main(["stats", "--fixture", "planted", "--data-dir", "x"])


# -- the score-test-once guard on real data ---------------------------------------
@pytest.fixture
def real_dir(tmp_path, monkeypatch):
    root = write_fixture(
        _real_make_fixture("planted", n_per_source=12, captions_per_image=2, dim=8),
        tmp_path / "data",
    )
    monkeypatch.chdir(tmp_path)
    return root


def test_eval_defaults_to_dev(capsys):
    main(["eval", "--fixture", "planted", "--resamples", "50"])
    assert "split=dev" in capsys.readouterr().out


def test_dev_on_real_data_writes_nothing_unless_asked(real_dir, tmp_path, capsys):
    assert main(["eval", "--data-dir", str(real_dir), "--resamples", "50"]) == 0
    assert not (tmp_path / "results").exists()


def test_test_split_on_real_data_writes_the_fixed_path_once(real_dir, tmp_path, capsys):
    args = ["eval", "--data-dir", str(real_dir), "--split", "test", "--resamples", "50"]
    assert main(args) == 0
    written = tmp_path / "results" / "test" / "eval.json"
    assert json.loads(written.read_text(encoding="utf-8"))["config"]["split"] == "test"
    before = written.read_text(encoding="utf-8")

    assert main(args) == 2
    err = capsys.readouterr().err
    assert "already scored" in err and "EVAL_PLAN.md" in err
    assert written.read_text(encoding="utf-8") == before


def test_test_split_on_real_data_refuses_a_custom_out_and_all(real_dir, tmp_path, capsys):
    base = ["eval", "--data-dir", str(real_dir), "--resamples", "50"]
    assert main([*base, "--split", "test", "--out", str(tmp_path / "x.json")]) == 2
    assert main([*base, "--split", "all"]) == 2
    assert not (tmp_path / "results").exists()


def test_fixture_runs_are_exempt_from_the_guard(tmp_path, capsys):
    out = tmp_path / "a.json"
    args = [
        "eval",
        "--fixture",
        "planted",
        "--split",
        "test",
        "--resamples",
        "50",
        "--out",
        str(out),
    ]
    assert main(args) == 0
    assert main(args) == 0
    assert main([*args[:-2], "--out", str(out)]) == 0
    assert main(["eval", "--fixture", "planted", "--split", "all", "--resamples", "50"]) == 0
