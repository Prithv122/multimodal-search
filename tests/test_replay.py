"""Replay: a saved run reproduces from its stored config, and tampering is caught."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mmsearch import cli
from mmsearch.cli import main
from mmsearch.fixtures import make_fixture, write_fixture, write_tags_csv


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A data directory and one saved dev run (results JSON + per-query CSV)."""
    monkeypatch.chdir(tmp_path)
    data = write_fixture(
        make_fixture("filler", n_per_source=20, captions_per_image=2, dim=16), tmp_path / "data"
    )
    results = tmp_path / "out" / "dev.json"
    assert main(["eval", "--data-dir", str(data), "--resamples", "300", "--out", str(results)]) == 0
    return data, results, results.with_name("dev_per_query.csv")


def _args(data: Path, results: Path, *extra: str) -> list[str]:
    return ["replay", "--data-dir", str(data), "--results", str(results), *extra]


def _tree(root: Path) -> dict[str, float]:
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_replay_of_a_fresh_run_passes_quietly(run, capsys, tmp_path):
    data, results, per_query = run
    capsys.readouterr()
    before = _tree(tmp_path)
    assert main(_args(data, results)) == 0
    out = capsys.readouterr().out
    n_rows = len(json.loads(results.read_text(encoding="utf-8"))["rows"])
    n_pq = len(list(csv.DictReader(per_query.open(encoding="utf-8"))))
    assert out.strip() == f"replay OK: {n_rows} rows, {n_pq} per-query rows match {results}"
    assert "nDCG" not in out and "H1" not in out  # no metric table, no verdicts
    assert _tree(tmp_path) == before  # nothing created or rewritten


def test_tags_added_after_a_run_do_not_break_its_replay(run, capsys, tmp_path):
    # Like the real data at scoring time: no tags.csv and every query's tag null.
    data, _, _ = run
    (data / "tags.csv").unlink(missing_ok=True)
    queries_path = data / "queries.jsonl"
    rows = [json.loads(line) for line in queries_path.open(encoding="utf-8")]
    queries_path.write_text(
        "".join(json.dumps({**r, "tag": None}) + "\n" for r in rows), encoding="utf-8"
    )
    untagged = tmp_path / "out" / "untagged.json"
    assert (
        main(["eval", "--data-dir", str(data), "--resamples", "300", "--out", str(untagged)]) == 0
    )
    write_tags_csv(data / "tags.csv", {r["qid"]: "text" for r in rows[:10]})
    capsys.readouterr()
    assert main(_args(data, untagged)) == 0
    assert capsys.readouterr().out.startswith("replay OK")


def test_a_tampered_mean_fails_with_exit_1_and_a_diff(run, capsys):
    data, results, _ = run
    payload = json.loads(results.read_text(encoding="utf-8"))
    payload["rows"][3]["mean"] += 0.01
    results.write_text(json.dumps(payload), encoding="utf-8")
    capsys.readouterr()
    assert main(_args(data, results)) == 1
    err = capsys.readouterr().err
    assert "MISMATCH" in err and "mean" in err and "(stored)" in err


def test_a_tampered_interval_fails_in_full_mode_but_not_fast_mode(run, capsys):
    data, results, _ = run
    payload = json.loads(results.read_text(encoding="utf-8"))
    payload["rows"][0]["ci_hi"] += 0.01
    results.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_args(data, results)) == 1
    capsys.readouterr()
    assert main(_args(data, results, "--resamples", "50")) == 0
    assert "fast" in capsys.readouterr().out


def test_fast_mode_still_catches_means_checks_and_per_query_rows(run, capsys):
    data, results, per_query = run
    good_results = results.read_text(encoding="utf-8")
    payload = json.loads(good_results)
    payload["rows"][2]["mean"] += 0.01
    results.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_args(data, results, "--resamples", "50")) == 1

    payload = json.loads(good_results)
    payload["checks"]["best_fusion"] = (
        "fuse_rrf" if payload["checks"]["best_fusion"] != "fuse_rrf" else "fuse_score"
    )
    results.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_args(data, results, "--resamples", "50")) == 1

    results.write_text(good_results, encoding="utf-8")
    rows = list(csv.DictReader(per_query.open(encoding="utf-8")))
    rows[5]["clip:ndcg@10"] = "0.123456"
    with per_query.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    capsys.readouterr()
    assert main(_args(data, results, "--resamples", "50")) == 1
    assert "per_query[" in capsys.readouterr().err


def test_a_changed_config_value_is_caught(run):
    data, results, _ = run
    payload = json.loads(results.read_text(encoding="utf-8"))
    payload["config"]["weight"]["chosen_clip_weight"] = 0.123
    results.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_args(data, results)) == 1


def test_replay_uses_the_stored_config_not_the_defaults(run, tmp_path, capsys):
    data, _, _ = run
    results = tmp_path / "custom.json"
    argv = ["eval", "--data-dir", str(data), "--resamples", "120", "--top-k", "50"]
    argv += ["--dev-fraction", "0.3", "--split-seed", "5", "--seed", "9", "--out", str(results)]
    argv += ["--reading-min-gain", "0.2", "--arm", "clip", "fuse_rrf"]
    assert main(argv) == 0
    capsys.readouterr()
    assert main(_args(data, results)) == 0


def test_replay_of_the_guarded_test_record(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data = write_fixture(
        make_fixture("planted", n_per_source=12, captions_per_image=2, dim=8), tmp_path / "data"
    )
    assert main(["eval", "--data-dir", str(data), "--split", "test", "--resamples", "80"]) == 0
    capsys.readouterr()
    assert main(["replay", "--data-dir", str(data), "--resamples", "40"]) == 0  # default path
    assert main(["replay", "--data-dir", str(data)]) == 0
    assert (tmp_path / "results" / "test" / "per_query.csv").exists()


@pytest.mark.parametrize(
    "case", ["missing_results", "bad_json", "wrong_shape", "no_csv", "no_data"]
)
def test_bad_input_exits_2(run, tmp_path, capsys, case):
    data, results, per_query = run
    if case == "missing_results":
        results = tmp_path / "nope.json"
    elif case == "bad_json":
        results.write_text("{not json", encoding="utf-8")
    elif case == "wrong_shape":
        results.write_text("[1, 2]", encoding="utf-8")
    elif case == "no_csv":
        per_query.unlink()
    elif case == "no_data":
        data = tmp_path / "absent"
    assert main(_args(data, results)) == 2
    assert "error:" in capsys.readouterr().err


def test_results_missing_a_config_key_is_bad_input(run, capsys):
    data, results, _ = run
    payload = json.loads(results.read_text(encoding="utf-8"))
    del payload["config"]["top_k"]
    results.write_text(json.dumps(payload), encoding="utf-8")
    assert main(_args(data, results)) == 2
    assert "top_k" in capsys.readouterr().err


def test_replay_command_is_registered():
    assert "replay" in cli.build_parser().format_help()
