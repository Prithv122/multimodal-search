"""Turn the filled blind tagging sheet into ``data/tags.csv``.

The owner tags ``data/tagging/tag_sheet.csv`` by its blind ids (``t001``..``t100``), which hide
the source. ``data/tagging/tag_key.csv`` maps each blind id back to a real query id. This script
joins the two, checks every label, and writes ``data/tags.csv`` (``qid,tag``), the file the
gallery loader and ``mmsearch tagged`` read.

It refuses, rather than guesses, when the sheet is incomplete, a label is unknown, the sheet and
the key disagree, or a qid is not in ``data/queries.jsonl``. ``--allow-partial`` writes only the
tagged rows, for a first look before the sheet is done; the README table uses a complete sheet.

    uv run python scripts/import_tags.py              # write data/tags.csv
    uv run python scripts/import_tags.py --check      # validate only, write nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

TAGS = ("visual", "text", "both")


class TagError(ValueError):
    """The sheet or key cannot be turned into tags.csv as it stands."""


def _read_csv(path: Path, columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise TagError(f"{path} not found")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = columns - set(reader.fieldnames or ())
        if missing:
            raise TagError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def convert(
    sheet_path: Path, key_path: Path, queries_path: Path, *, allow_partial: bool = False
) -> dict[str, str]:
    """Return ``{qid: tag}`` from the filled sheet and its key, or raise ``TagError``."""
    sheet = _read_csv(sheet_path, {"qid", "tag"})
    key_rows = _read_csv(key_path, {"blind_id", "qid"})
    key = {r["blind_id"].strip(): r["qid"].strip() for r in key_rows}
    if len(key) != len(key_rows):
        raise TagError(f"{key_path}: duplicate blind ids")

    blind_ids = [r["qid"].strip() for r in sheet]
    if len(set(blind_ids)) != len(blind_ids):
        raise TagError(f"{sheet_path}: duplicate ids in the qid column")
    if set(blind_ids) != set(key):
        extra, absent = set(blind_ids) - set(key), set(key) - set(blind_ids)
        raise TagError(
            f"sheet and key disagree: not in key {sorted(extra)[:3]}, "
            f"missing from sheet {sorted(absent)[:3]}"
        )

    tags: dict[str, str] = {}
    blank: list[str] = []
    for row, blind in zip(sheet, blind_ids, strict=True):
        label = (row["tag"] or "").strip().lower()
        if not label:
            blank.append(blind)
            continue
        if label not in TAGS:
            raise TagError(f"{sheet_path}: {blind} has tag {row['tag']!r}; use one of {TAGS}")
        tags[key[blind]] = label
    if blank and not allow_partial:
        raise TagError(
            f"{len(blank)} of {len(sheet)} rows are untagged (e.g. {blank[:3]}); "
            "finish the sheet or pass --allow-partial"
        )

    with queries_path.open(encoding="utf-8") as handle:
        known = {json.loads(line)["qid"] for line in handle if line.strip()}
    unknown = sorted(set(tags) - known)
    if unknown:
        raise TagError(f"qids not in {queries_path}: {unknown[:3]}")
    return tags


def write_tags(path: Path, tags: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "tag"])
        writer.writerows(sorted(tags.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--allow-partial", action="store_true", help="write only tagged rows")
    parser.add_argument("--check", action="store_true", help="validate only; write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite an existing tags.csv")
    args = parser.parse_args(argv)

    tagging = args.data_dir / "tagging"
    out = args.data_dir / "tags.csv"
    try:
        tags = convert(
            tagging / "tag_sheet.csv",
            tagging / "tag_key.csv",
            args.data_dir / "queries.jsonl",
            allow_partial=args.allow_partial,
        )
        if not tags:
            raise TagError("no rows are tagged yet")
        if not args.check and out.exists() and not args.force:
            raise TagError(f"{out} exists; pass --force to replace it")
    except TagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    counts = Counter(tags.values())
    summary = ", ".join(f"{t}={counts.get(t, 0)}" for t in TAGS)
    if args.check:
        print(f"sheet OK: {len(tags)} tagged ({summary}); nothing written")
        return 0
    write_tags(out, tags)
    print(f"wrote {out}: {len(tags)} tags ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
