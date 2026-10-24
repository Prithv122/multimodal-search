"""Command line: ``mmsearch stats | eval | search``."""

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
from mmsearch.report import format_stats, format_study
from mmsearch.split import DEFAULT_DEV_FRACTION, DEFAULT_SPLIT_SEED, split_images


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
    ev.add_argument("--split", choices=SPLITS, default="test")
    ev.add_argument("--arm", nargs="+", choices=ARMS, help="only show these arms")
    ev.add_argument("--resamples", type=int, default=20_000)
    ev.add_argument("--seed", type=int, default=EvalConfig.seed, help="bootstrap seed")
    ev.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    ev.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    ev.add_argument("--top-k", type=int, default=EvalConfig.top_k)
    ev.add_argument("--min-confidence", type=float, default=None, help="OCR confidence floor")
    ev.add_argument("--reading-min-gain", type=float, default=EvalConfig.reading_min_gain)
    ev.add_argument("--visual-max-diff", type=float, default=EvalConfig.visual_max_abs_diff)
    ev.add_argument("--out", type=Path, help="write results JSON here")
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
    return parser


def _cmd_stats(args: argparse.Namespace) -> int:
    gallery = _gallery(args)
    print(format_stats(gallery.stats()))
    assignment = split_images(gallery.items)
    n_dev = sum(1 for s in assignment.values() if s == "dev")
    print(f"split (seed {DEFAULT_SPLIT_SEED}): dev {n_dev} images / test {len(assignment) - n_dev}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
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
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"stats": _cmd_stats, "eval": _cmd_eval, "search": _cmd_search}
    try:
        return handlers[args.command](args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
