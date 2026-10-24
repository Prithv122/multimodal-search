"""Plain-text rendering of a :class:`~mmsearch.evaluate.StudyResult`."""

from __future__ import annotations

from typing import Any

from mmsearch.evaluate import METRICS, StudyResult

_LABELS = {
    "all": "all queries",
    "source=coco": "visual slice (COCO)",
    "source=textcaps": "reading slice (TextCaps)",
    "tag=visual": "hand-tagged: visual",
    "tag=text": "hand-tagged: text",
    "tag=both": "hand-tagged: both",
}


def format_stats(stats: dict[str, Any]) -> str:
    lines = [f"images:  {stats['images']}", f"queries: {stats['queries']}"]
    for source, counts in stats["per_source"].items():
        lines.append(f"  {source:<9} images={counts['images']:<5} queries={counts['queries']}")
    tagged = stats["tagged_queries"]
    lines.append("hand-tagged queries: " + ", ".join(f"{t}={n}" for t, n in tagged.items()))
    lines.append(
        f"OCR: {stats['ocr_records']} records, {stats['images_with_ocr_text']} images with text"
    )
    if stats["embedding_dim"] is None:
        lines.append("embeddings: none")
    else:
        lines.append(f"embeddings: dim={stats['embedding_dim']} model={stats['embedding_model']}")
    return "\n".join(lines)


def _check_line(name: str, check: dict[str, Any]) -> str:
    if check["status"] == "N/A":
        return f"{name}: N/A ({check['reason']})"
    interval = f"[{check['ci_lo']:+.3f}, {check['ci_hi']:+.3f}]"
    if name.startswith("H1"):
        detail = (
            f"{check['best_fusion']} - clip = {check['diff_ndcg@10']:+.3f} nDCG@10 {interval}; "
            f"need >= {check['min_gain']:.2f} and interval excluding 0"
        )
    else:
        detail = (
            f"{check['best_fusion']} - clip = {check['diff_ndcg@10']:+.3f} nDCG@10 {interval}; "
            f"need |diff| <= {check['max_abs_diff']:.2f}"
        )
    return f"{name}: {check['status']}  ({detail})"


def format_study(result: StudyResult) -> str:
    cfg, rows, checks = result.config, result.rows, result.checks
    out = [
        f"split={cfg['split']}  images={cfg['n_images']} (dev {cfg['n_dev_images']} / "
        f"test {cfg['n_test_images']})  queries={cfg['n_queries']}  "
        f"resamples={cfg['resamples']}  seed={cfg['seed']}",
        f"fuse_score_weighted: CLIP weight {cfg['weight']['chosen_clip_weight']:.2f} "
        f"(chosen on dev, dev nDCG@10 {cfg['weight']['dev_ndcg@10']:.3f})",
        f"best fusion (chosen on dev): {checks['best_fusion']}  [dev nDCG@10 "
        + ", ".join(f"{k} {v:.3f}" for k, v in checks["best_fusion_dev_ndcg@10"].items())
        + "; the weighted arm's weight was tuned on dev]",
        "MRR@100: reciprocal rank within the top 100 (gold outside it counts 0)",
    ]
    if not checks["confirmatory"]:
        out.append(
            f"NOTE: split={cfg['split']} is not the held-out test split; "
            "weight and best fusion were chosen on dev, so these numbers are not confirmatory."
        )

    slices = list(dict.fromkeys(r["slice"] for r in rows))
    for name in slices:
        arm_rows = [r for r in rows if r["kind"] == "arm" and r["slice"] == name]
        if not arm_rows:
            continue
        first = arm_rows[0]
        out.append("")
        out.append(
            f"== {name}  {_LABELS.get(name, '')}  "
            f"[{first['n_images']} images, {first['n_queries']} queries]"
        )
        head = f"{'arm':<20}" + "".join(f"{m:>10}" for m in METRICS) + "   nDCG@10 95% CI"
        out.append(head)
        for arm in dict.fromkeys(r["arm"] for r in arm_rows):
            cells = {r["metric"]: r for r in arm_rows if r["arm"] == arm}
            nd = cells["ndcg@10"]
            out.append(
                f"{arm:<20}"
                + "".join(f"{cells[m]['mean']:>10.3f}" for m in METRICS)
                + f"   [{nd['ci_lo']:.3f}, {nd['ci_hi']:.3f}]"
            )
        diffs = [
            r
            for r in rows
            if r["kind"] == "diff" and r["slice"] == name and r["metric"] == "ndcg@10"
        ]
        if diffs:
            out.append("paired nDCG@10 differences (same resamples):")
            for r in diffs:
                out.append(
                    f"  {r['arm_a'] + ' - ' + r['arm_b']:<32} {r['mean']:+.3f}  "
                    f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
                )

    out.append("")
    out.append("Pre-registered checks" + ("" if checks["confirmatory"] else " (NOT confirmatory)"))
    out.append("  " + _check_line("H1 reading", checks["H1_reading"]))
    out.append("  " + _check_line("H2 visual ", checks["H2_visual"]))
    for name, sec in checks["best_fusion_vs_best_single"].items():
        verdict = "beats" if sec["fusion_beats_best_single"] else "does not beat"
        out.append(
            f"  {name}: {checks['best_fusion']} {verdict} best single arm "
            f"({sec['best_single_arm_on_dev']}): {sec['diff_ndcg@10']:+.3f} "
            f"[{sec['ci_lo']:+.3f}, {sec['ci_hi']:+.3f}]"
        )
    return "\n".join(out)
