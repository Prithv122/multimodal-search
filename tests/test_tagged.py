"""The tagged-subset table regroups saved per-query results; it never re-ranks."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from mmsearch.cli import main
from mmsearch.evaluate import EvalConfig, run_study
from mmsearch.fixtures import make_fixture
from mmsearch.gallery import DataError
from mmsearch.tagged import read_per_query, tagged_table, write_per_query


@pytest.fixture(scope="module")
def planted_result():
    return run_study(make_fixture("planted"), EvalConfig(split="test", resamples=200))


def test_per_query_rows_cover_the_scored_split_only(planted_result) -> None:
    rows = planted_result.per_query
    assert rows
    assert {r["split"] for r in rows} == {"test"}
    assert len({r["qid"] for r in rows}) == len(rows)


def test_per_query_means_reproduce_the_main_table(planted_result) -> None:
    rows = planted_result.per_query
    for r in planted_result.rows:
        if r["kind"] == "arm" and r["slice"] == "source=textcaps" and r["metric"] == "ndcg@10":
            values = [q[f"{r['arm']}:ndcg@10"] for q in rows if q["source"] == "textcaps"]
            assert np.mean(values) == pytest.approx(r["mean"])


def test_round_trip_and_exclusive_write(planted_result, tmp_path) -> None:
    path = tmp_path / "per_query.csv"
    write_per_query(path, planted_result.per_query, exclusive=True)
    back = read_per_query(path)
    assert len(back) == len(planted_result.per_query)
    assert back[0]["clip:ndcg@10"] == pytest.approx(planted_result.per_query[0]["clip:ndcg@10"])
    with pytest.raises(FileExistsError):
        write_per_query(path, planted_result.per_query, exclusive=True)


def test_tagged_table_groups_by_tag_and_counts_missing(planted_result) -> None:
    rows = planted_result.per_query
    text_qids = [r["qid"] for r in rows if r["source"] == "textcaps"][:30]
    visual_qids = [r["qid"] for r in rows if r["source"] == "coco"][:30]
    tags = {q: "text" for q in text_qids} | {q: "visual" for q in visual_qids}
    tags["not-a-scored-qid"] = "both"
    best = planted_result.checks["best_fusion"]
    table = {e["tag"]: e for e in tagged_table(rows, tags, best_fusion=best, resamples=300)}
    assert table["text"]["n_queries"] == 30
    assert table["both"]["n_queries"] == 0
    assert table["both"]["not_in_split"] == 1
    # The planted effect lives in the reading slice, so it shows under the "text" tag.
    assert table["text"]["diff"]["ci_lo"] > 0
    assert abs(table["visual"]["diff"]["mean"]) <= 0.02


def test_tagged_rejects_an_arm_not_in_the_results(planted_result) -> None:
    with pytest.raises(DataError):
        tagged_table(planted_result.per_query, {}, best_fusion="no_such_arm")


def test_cli_eval_then_tagged(tmp_path, capsys) -> None:
    out = tmp_path / "eval.json"
    assert (
        main(
            [
                "eval",
                "--fixture",
                "planted",
                "--split",
                "test",
                "--resamples",
                "200",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    per_query = tmp_path / "eval_per_query.csv"
    assert per_query.exists()
    with per_query.open(encoding="utf-8", newline="") as handle:
        qids = [r["qid"] for r in csv.DictReader(handle)][:10]
    tags = tmp_path / "tags.csv"
    tags.write_text("qid,tag\n" + "".join(f"{q},text\n" for q in qids), encoding="utf-8")
    table_out = tmp_path / "tagged.json"
    code = main(
        [
            "tagged",
            "--results",
            str(out),
            "--tags",
            str(tags),
            "--resamples",
            "200",
            "--out",
            str(table_out),
        ]
    )
    assert code == 0
    payload = json.loads(table_out.read_text(encoding="utf-8"))
    assert (
        payload["best_fusion"]
        == json.loads(out.read_text(encoding="utf-8"))["checks"]["best_fusion"]
    )
    assert "tag=text" in capsys.readouterr().out
