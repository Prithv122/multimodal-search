"""The tagged-subset table: the owner's hand tags applied to an already-scored split.

The slice in the main table is the source dataset, not the query type. The owner tags a sample
of queries as visual / text / both, and this module regroups the saved per-query metrics of a
scored split by those tags. It ranks nothing and chooses nothing, so applying tags after the
test split was scored is not a second look at test: the rankings, the weight and the best
fusion are the ones already on disk.
"""

from __future__ import annotations

import csv
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mmsearch.evaluate import cluster_bootstrap_means, percentile_interval
from mmsearch.gallery import TAGS, DataError, load_tags

PER_QUERY_FILE = "per_query.csv"


def write_per_query(path: Path, rows: Sequence[Mapping[str, Any]], *, exclusive: bool) -> None:
    """Write the per-query rows as CSV; ``exclusive`` refuses to overwrite (the test record)."""
    if not rows:
        raise DataError("no per-query rows to write")
    fields = list(rows[0])
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_per_query(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataError(f"{path} not found; score a split with `mmsearch eval` first")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if ":" in key:
                row[key] = float(value)
    return rows


def tagged_table(
    per_query: Sequence[Mapping[str, Any]],
    tags: Mapping[str, str],
    *,
    best_fusion: str,
    metric: str = "ndcg@10",
    resamples: int = 20_000,
    seed: int = 47,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """One row per tag: each arm's mean and the paired ``best_fusion - clip`` difference.

    Intervals use the same cluster bootstrap as the main table (images resampled with
    replacement). Tagged qids that are not in ``per_query`` (for example, tagged images that
    fell in the other split) are skipped and counted.
    """
    arms = sorted({key.split(":", 1)[0] for key in per_query[0] if ":" in key})
    for needed in ("clip", best_fusion):
        if needed not in arms:
            raise DataError(f"arm {needed!r} is not in the per-query results")
    by_qid = {row["qid"]: row for row in per_query}
    out: list[dict[str, Any]] = []
    for tag in TAGS:
        rows = [by_qid[q] for q, t in tags.items() if t == tag and q in by_qid]
        missing = sum(1 for q, t in tags.items() if t == tag and q not in by_qid)
        entry: dict[str, Any] = {
            "tag": tag,
            "metric": metric,
            "n_queries": len(rows),
            "n_images": len({r["image_id"] for r in rows}),
            "not_in_split": missing,
        }
        if not rows:
            out.append(entry)
            continue
        values = np.array([[r[f"{a}:{metric}"] for a in arms] for r in rows])
        clusters = np.array([r["image_id"] for r in rows])
        boot = cluster_bootstrap_means(
            values, clusters, resamples=resamples, seed=[seed, zlib.crc32(tag.encode())]
        )
        lo, hi = percentile_interval(boot, alpha)
        entry["arms"] = {
            a: {"mean": float(values[:, i].mean()), "ci_lo": float(lo[i]), "ci_hi": float(hi[i])}
            for i, a in enumerate(arms)
        }
        ib, ic = arms.index(best_fusion), arms.index("clip")
        dlo, dhi = percentile_interval(boot[:, ib] - boot[:, ic], alpha)
        entry["diff"] = {
            "arm_a": best_fusion,
            "arm_b": "clip",
            "mean": float(values[:, ib].mean() - values[:, ic].mean()),
            "ci_lo": float(dlo),
            "ci_hi": float(dhi),
        }
        out.append(entry)
    return out


def format_tagged(table: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for entry in table:
        head = (
            f"tag={entry['tag']:<7} queries={entry['n_queries']:<4} images={entry['n_images']:<4}"
            f" (not in split: {entry['not_in_split']})"
        )
        lines.append(head)
        for arm, v in entry.get("arms", {}).items():
            band = f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
            lines.append(f"    {arm:<20} {entry['metric']} {v['mean']:.3f} {band}")
        if "diff" in entry:
            d = entry["diff"]
            band = f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"
            lines.append(f"    {d['arm_a']} - {d['arm_b']}: {d['mean']:+.3f} {band}")
    return "\n".join(lines)


def load_tag_file(path: Path) -> dict[str, str]:
    """Tags as ``{qid: tag}``, blanks dropped."""
    return {q: t for q, t in load_tags(path).items() if t}
