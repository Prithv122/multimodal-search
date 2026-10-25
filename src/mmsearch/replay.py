"""Replay a saved run in memory and check that it reproduces the stored files.

``replay`` rebuilds the study from the config stored in a results JSON (split, seeds, resamples,
top_k, OCR floor, thresholds, weight grid, dev fraction) and the data directory, then compares
the outcome with what is on disk. It writes nothing and prints no metric table: the point is to
show that the committed numbers come from the committed code and arrays, not to look at them
again. The test split is therefore not "scored a second time"; its record is only re-derived.

Two modes:

full
    The stored resamples. Every row's ``mean``, ``ci_lo`` and ``ci_hi`` (within ``TOLERANCE``),
    the config and weight sweep, the checks and the per-query CSV must all match.
fast (``resamples`` override)
    Fewer resamples, so the intervals change on purpose and are not compared. What is still
    compared: every row's ``mean``, the config (except ``resamples``), the checks without their
    interval-derived fields (``ci_lo``, ``ci_hi``, ``ci_excludes_zero``,
    ``fusion_beats_best_single`` and H1's ``status``, which needs the interval), and the
    per-query CSV.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmsearch.evaluate import EvalConfig, run_study
from mmsearch.gallery import DataError, load_gallery
from mmsearch.tagged import read_per_query

TOLERANCE = 1e-9
MAX_REPORTED = 20
_INTERVAL_KEYS = {"ci_lo", "ci_hi", "ci_excludes_zero", "fusion_beats_best_single"}


@dataclass
class ReplayReport:
    path: Path
    n_rows: int
    n_per_query: int
    fast: bool
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        if self.ok:
            mode = " (fast: point estimates only, intervals not compared)" if self.fast else ""
            return (
                f"replay OK{mode}: {self.n_rows} rows, {self.n_per_query} per-query rows "
                f"match {self.path}"
            )
        shown = self.mismatches[:MAX_REPORTED]
        more = len(self.mismatches) - len(shown)
        lines = [f"replay MISMATCH: {len(self.mismatches)} difference(s) against {self.path}"]
        lines += [f"  {m}" for m in shown]
        if more > 0:
            lines.append(f"  ... and {more} more")
        return "\n".join(lines)


def config_from_results(stored: dict[str, Any], resamples: int | None = None) -> EvalConfig:
    """Rebuild the :class:`EvalConfig` that produced a results file."""
    try:
        arms = stored["arms"]
        return EvalConfig(
            split=stored["split"],
            arms=tuple(arms) if arms else None,
            resamples=int(resamples if resamples is not None else stored["resamples"]),
            seed=int(stored["seed"]),
            split_seed=int(stored["split_seed"]),
            dev_fraction=float(stored["dev_fraction"]),
            top_k=int(stored["top_k"]),
            min_confidence=stored["min_confidence"],
            weight_grid=tuple(float(w) for w in stored["weight_grid"]),
            reading_min_gain=float(stored["reading_min_gain"]),
            visual_max_abs_diff=float(stored["visual_max_abs_diff"]),
            alpha=float(stored["alpha"]),
        )
    except KeyError as exc:
        raise DataError(f"results config is missing {exc.args[0]!r}; cannot replay") from None


def _compare(a: Any, b: Any, path: str, out: list[str], skip: set[str] = frozenset()) -> None:
    """Deep comparison collecting readable differences; floats within ``TOLERANCE``."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key in skip:
                continue
            if key not in a:
                out.append(f"{path}.{key}: missing in stored file")
            elif key not in b:
                out.append(f"{path}.{key}: missing in replay")
            else:
                _compare(a[key], b[key], f"{path}.{key}", out, skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} (stored) vs {len(b)} (replay)")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _compare(x, y, f"{path}[{i}]", out, skip)
    elif (
        isinstance(a, int | float)
        and isinstance(b, int | float)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=TOLERANCE):
            out.append(f"{path}: {a!r} (stored) vs {b!r} (replay)")
    elif a != b:
        out.append(f"{path}: {a!r} (stored) vs {b!r} (replay)")


def _drop_interval_fields(checks: dict[str, Any]) -> dict[str, Any]:
    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in _INTERVAL_KEYS}
        return node

    cleaned = strip(checks)
    # H1 passes only if the interval excludes 0, so its verdict depends on the resamples.
    if isinstance(cleaned.get("H1_reading"), dict):
        cleaned["H1_reading"].pop("status", None)
    return cleaned


def replay(
    results_path: Path,
    per_query_path: Path,
    data_dir: Path,
    *,
    resamples: int | None = None,
) -> ReplayReport:
    """Recompute the study in memory and compare it with the stored files. Writes nothing."""
    if not results_path.is_file():
        raise DataError(f"{results_path} not found")
    try:
        stored = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{results_path}: invalid JSON ({exc.msg})") from exc
    if not isinstance(stored, dict) or not {"config", "rows", "checks"} <= set(stored):
        raise DataError(f"{results_path}: expected an object with config, rows and checks")
    stored_per_query = read_per_query(per_query_path)

    config = config_from_results(stored["config"], resamples)
    result = run_study(load_gallery(data_dir), config)
    fresh = json.loads(json.dumps(result.to_json()))
    fast = resamples is not None and resamples != stored["config"]["resamples"]

    mismatches: list[str] = []
    config_skip = {"resamples"} if fast else set()
    _compare(stored["config"], fresh["config"], "config", mismatches, config_skip)

    if len(stored["rows"]) != len(fresh["rows"]):
        mismatches.append(f"rows: {len(stored['rows'])} (stored) vs {len(fresh['rows'])} (replay)")
    else:
        row_skip = _INTERVAL_KEYS if fast else set()
        for i, (old, new) in enumerate(zip(stored["rows"], fresh["rows"], strict=True)):
            label = (
                f"rows[{i}] {old.get('slice')}/{old.get('arm') or old.get('arm_a')}"
                f"{'-' + old['arm_b'] if 'arm_b' in old else ''}/{old.get('metric')}"
            )
            _compare(old, new, label, mismatches, row_skip)

    if fast:
        _compare(
            _drop_interval_fields(stored["checks"]),
            _drop_interval_fields(fresh["checks"]),
            "checks",
            mismatches,
        )
    else:
        _compare(stored["checks"], fresh["checks"], "checks", mismatches)

    fresh_per_query = json.loads(json.dumps(result.per_query))
    if len(stored_per_query) != len(fresh_per_query):
        mismatches.append(
            f"per-query rows: {len(stored_per_query)} (stored) vs {len(fresh_per_query)} (replay)"
        )
    else:
        for old, new in zip(stored_per_query, fresh_per_query, strict=True):
            _compare(old, new, f"per_query[{old.get('qid')}]", mismatches)
            if len(mismatches) > 10 * MAX_REPORTED:
                break

    return ReplayReport(
        path=results_path,
        n_rows=len(stored["rows"]),
        n_per_query=len(stored_per_query),
        fast=fast,
        mismatches=mismatches,
    )
