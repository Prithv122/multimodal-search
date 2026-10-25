"""Command line: ``mmsearch stats | eval | search | tagged | replay``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mmsearch.arms import ARMS, Retriever
from mmsearch.embed import FakeEmbedder, load_encoder
from mmsearch.evaluate import SPLITS, EvalConfig, run_study
from mmsearch.fixtures import FIXTURE_KINDS, make_fixture
from mmsearch.gallery import DataError, Gallery, load_gallery
from mmsearch.replay import replay
from mmsearch.report import format_stats, format_study
from mmsearch.split import DEFAULT_DEV_FRACTION, DEFAULT_SPLIT_SEED, split_images
from mmsearch.tagged import (
    PER_QUERY_FILE,
    format_tagged,
    load_tag_file,
    read_per_query,
    tagged_table,
    write_per_query,
)

TEST_RESULT_PATH = Path("results") / "test" / "eval.json"


def _add_source(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data-dir", default="data", help="data directory (default: data)")
    group.add_argument(
        "--fixture", choices=FIXTURE_KINDS, help="use a synthetic gallery instead of --data-dir"
    )


def _gallery(args: argparse.Namespace) -> Gallery:
    if args.fixture:
        return make_fixture(args.fixture)
    return load_gallery(args.data_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmsearch", description="Text-to-image search: CLIP vs OCR-BM25 vs fusion."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="print counts for a data directory or fixture")
    _add_source(stats)

    ev = sub.add_parser("eval", help="score the arms and print the table")
    _add_source(ev)
    ev.add_argument(
        "--split",
        choices=SPLITS,
        default="dev",
        help="dev (default), or test to score the held-out split once (see EVAL_PLAN.md)",
    )
    ev.add_argument("--arm", nargs="+", choices=ARMS, help="only show these arms")
    ev.add_argument("--resamples", type=int, default=20_000)
    ev.add_argument("--seed", type=int, default=EvalConfig.seed, help="bootstrap seed")
    ev.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    ev.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    ev.add_argument("--top-k", type=int, default=EvalConfig.top_k)
    ev.add_argument("--min-confidence", type=float, default=None, help="OCR confidence floor")
    ev.add_argument("--reading-min-gain", type=float, default=EvalConfig.reading_min_gain)
    ev.add_argument("--visual-max-diff", type=float, default=EvalConfig.visual_max_abs_diff)
    ev.add_argument(
        "--out",
        type=Path,
        help=f"write results JSON here (real-data test runs always use {TEST_RESULT_PATH})",
    )
    ev.add_argument(
        "--fail-on-prereg", action="store_true", help="exit 1 when a pre-registered check fails"
    )

    se = sub.add_parser("search", help="rank the gallery for one text query")
    _add_source(se)
    se.add_argument("text", help="query text")
    se.add_argument("--arm", choices=ARMS, default="fuse_score_weighted")
    se.add_argument("--top-k", type=int, default=10)
    se.add_argument("--weight", type=float, default=0.5, help="CLIP weight for the weighted arm")
    se.add_argument("--query-qid", help="reuse the stored caption embedding of this qid")
    se.add_argument(
        "--encoder",
        help="'fake' or 'package.module:factory' returning an object with encode_query(text)",
    )

    tg = sub.add_parser(
        "tagged", help="regroup a scored split's per-query results by the owner's hand tags"
    )
    tg.add_argument(
        "--results",
        type=Path,
        default=TEST_RESULT_PATH,
        help=f"results JSON of a scored split; {PER_QUERY_FILE} must sit next to it "
        f"(default: {TEST_RESULT_PATH})",
    )
    tg.add_argument("--tags", type=Path, default=Path("data") / "tags.csv", help="qid,tag CSV")
    tg.add_argument("--resamples", type=int, default=20_000)
    tg.add_argument("--seed", type=int, default=EvalConfig.seed)
    tg.add_argument("--out", type=Path, help="write the table as JSON here")

    rp = sub.add_parser(
        "replay",
        help="recompute a saved run in memory and check it against the stored files",
    )
    rp.add_argument("--results", type=Path, default=TEST_RESULT_PATH, help="results JSON to check")
    rp.add_argument("--data-dir", type=Path, default=Path("data"))
    rp.add_argument(
        "--resamples",
        type=int,
        help="fast mode: fewer bootstrap resamples; compares means, checks without intervals and "
        "per-query rows, and skips the interval comparison",
    )
    return parser


def _cmd_stats(args: argparse.Namespace) -> int:
    gallery = _gallery(args)
    print(format_stats(gallery.stats()))
    assignment = split_images(gallery.items)
    n_dev = sum(1 for s in assignment.values() if s == "dev")
    print(f"split (seed {DEFAULT_SPLIT_SEED}): dev {n_dev} images / test {len(assignment) - n_dev}")
    return 0


def _test_guard(args: argparse.Namespace) -> Path | None:
    """Enforce score-test-once for real data; returns the fixed output path when it applies.

    Fixture runs are exempt and write wherever ``--out`` says. On real data, ``test`` always
    writes ``results/test/eval.json`` and refuses to run again once that file exists; ``all``
    is refused because it would score the test split without leaving that record.
    """
    if args.fixture or args.split == "dev":
        return None
    if args.split == "all":
        raise DataError("--split all would score the test split; use dev, or test (scored once)")
    if args.out is not None:
        raise DataError(f"test results are always written to {TEST_RESULT_PATH}; drop --out")
    if TEST_RESULT_PATH.exists():
        raise DataError(
            f"the test split was already scored ({TEST_RESULT_PATH} exists); per EVAL_PLAN.md "
            "it is scored once. Anything changed afterwards is a dated amendment, not a re-run"
        )
    return TEST_RESULT_PATH


def _cmd_eval(args: argparse.Namespace) -> int:
    test_path = _test_guard(args)
    gallery = _gallery(args)
    config = EvalConfig(
        split=args.split,
        arms=tuple(args.arm) if args.arm else None,
        resamples=args.resamples,
        seed=args.seed,
        split_seed=args.split_seed,
        dev_fraction=args.dev_fraction,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
        reading_min_gain=args.reading_min_gain,
        visual_max_abs_diff=args.visual_max_diff,
    )
    result = run_study(gallery, config)
    print(format_study(result))
    out = test_path or args.out
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result.to_json(), indent=2) + "\n"
        try:
            # Exclusive create for the guarded test file, so two runs cannot both write it.
            with out.open("x" if test_path else "w", encoding="utf-8") as handle:
                handle.write(payload)
        except FileExistsError:
            raise DataError(
                f"{out} appeared while scoring; the test split is scored once"
            ) from None
        # The guarded test record keeps the fixed name; any other run gets a sibling file.
        per_query_path = out.parent / PER_QUERY_FILE
        if test_path is None:
            per_query_path = out.with_name(out.stem + "_" + PER_QUERY_FILE)
        write_per_query(per_query_path, result.per_query, exclusive=test_path is not None)
        print(f"\nwrote {out} and {per_query_path}")
    if args.fail_on_prereg:
        failed = [k for k in ("H1_reading", "H2_visual") if result.checks[k]["status"] == "FAIL"]
        if failed:
            print(f"pre-registered check failed: {', '.join(failed)}", file=sys.stderr)
            return 1
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    gallery = _gallery(args)
    retriever = Retriever.from_gallery(gallery, top_k=max(args.top_k, 100))
    vector = None
    if args.arm != "ocr_bm25":
        if args.query_qid:
            try:
                vector = gallery.query_vector(args.query_qid)
            except KeyError:
                raise DataError(f"unknown qid {args.query_qid!r}") from None
        elif args.encoder:
            fallback = FakeEmbedder(gallery.require_embeddings().dim) if args.fixture else None
            vector = load_encoder(args.encoder, fallback=fallback).encode_query(args.text)
        elif args.fixture:
            vector = FakeEmbedder(gallery.require_embeddings().dim).encode_query(args.text)
        else:
            raise DataError(f"arm {args.arm!r} needs a query vector: pass --query-qid or --encoder")
    ranking = retriever.rank(
        args.arm, args.text, vector, weight=args.weight, query_key=args.query_qid
    )[: args.top_k]
    if not ranking:
        print("no results")
        return 0
    for rank, (image_id, score) in enumerate(ranking, start=1):
        item = gallery.item(image_id)
        print(f"{rank:>3}  {score:8.4f}  {image_id}  [{item.source}]  {item.url or item.file_name}")
    return 0


def _per_query_path(results: Path) -> Path:
    """Where the per-query CSV of a results file lives (fixed name for the guarded test file)."""
    if results.resolve() == TEST_RESULT_PATH.resolve():
        return results.parent / PER_QUERY_FILE
    return results.with_name(results.stem + "_" + PER_QUERY_FILE)


def _cmd_replay(args: argparse.Namespace) -> int:
    report = replay(
        args.results, _per_query_path(args.results), args.data_dir, resamples=args.resamples
    )
    if report.ok:
        print(report.summary())
        return 0
    print(report.summary(), file=sys.stderr)
    return 1


def _cmd_tagged(args: argparse.Namespace) -> int:
    if not args.results.exists():
        raise DataError(f"{args.results} not found; score a split first")
    results = json.loads(args.results.read_text(encoding="utf-8"))
    best_fusion = results["checks"]["best_fusion"]
    per_query_path = args.results.parent / PER_QUERY_FILE
    if args.results.resolve() != TEST_RESULT_PATH.resolve():
        per_query_path = args.results.with_name(args.results.stem + "_" + PER_QUERY_FILE)
    if not args.tags.exists():
        raise DataError(f"{args.tags} not found; the owner's hand tags are not in yet")
    table = tagged_table(
        read_per_query(per_query_path),
        load_tag_file(args.tags),
        best_fusion=best_fusion,
        resamples=args.resamples,
        seed=args.seed,
    )
    print(f"best fusion (chosen on dev): {best_fusion}")
    print(format_tagged(table))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"best_fusion": best_fusion, "results": str(args.results), "rows": table}
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "stats": _cmd_stats,
        "eval": _cmd_eval,
        "search": _cmd_search,
        "tagged": _cmd_tagged,
        "replay": _cmd_replay,
    }
    try:
        return handlers[args.command](args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
