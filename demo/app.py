"""Gradio demo: type a query, pick an arm, see the top images and the OCR text of each hit.

Run from the repository root::

    uv run --extra embed --extra demo python demo/app.py

Images are fetched by your browser from their original hosts (COCO and Open Images); none are
stored in the repository. Free-text queries need the CLIP text encoder (the ``embed`` extra and
a one-off weight download); the example captions use the stored embeddings and need neither.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `python demo/app.py` puts demo/, not the repo root, on the path
    sys.path.insert(0, str(ROOT))

from demo.core import (  # noqa: E402
    DEFAULT_TOP_K,
    DemoSearch,
    example_choices,
    gallery_items,
    load_demo_gallery,
    ocr_rows,
    recorded_weight,
)
from mmsearch.arms import ARMS  # noqa: E402

ARM_HELP = {
    "clip": "CLIP text-to-image cosine similarity",
    "ocr_bm25": "BM25 of the query against OCR text read from the images",
    "fuse_rrf": "reciprocal rank fusion of CLIP and OCR-BM25",
    "fuse_score": "equal-weight sum of normalised CLIP and OCR-BM25 scores",
    "fuse_score_weighted": "weighted sum of normalised scores (weight shown below)",
}
NO_EXAMPLE = "(none)"


def _encoder_factory():
    from demo.encoder import ClipTextEncoder

    return ClipTextEncoder(ROOT / "data")


def build_app(data_dir: Path, results_dir: Path, *, warm_up: bool = True):
    import gradio as gr

    gallery, notices = load_demo_gallery(data_dir)
    weight, weight_file = recorded_weight(results_dir)
    engine = DemoSearch(
        gallery,
        encoder_factory=_encoder_factory,
        weight=weight if weight is not None else 0.5,
        weight_source=f"results/dev/{weight_file}" if weight_file else None,
    )
    if warm_up:  # load CLIP in the background while the page opens
        threading.Thread(target=engine.warm_up, daemon=True).start()
    examples = example_choices(gallery.queries)
    by_qid = {q.qid: q.text for q in gallery.queries}
    choices = [(NO_EXAMPLE, ""), *examples]

    def run(text: str, arm: str, top_k: int):
        try:
            result = engine.search(text, arm, int(top_k))
        except ModuleNotFoundError as exc:
            raise gr.Error(
                f"Free-text queries need the CLIP text encoder ({exc}). Start the app with "
                "`uv run --extra embed --extra demo python demo/app.py`, "
                "or pick an example caption."
            ) from exc
        except (RuntimeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        status = f"**{result.arm_used}** - top {len(result.hits)} of {len(gallery.items)} images"
        if result.arm_used != arm:
            status = f"**{arm}** (shown as {result.arm_used}) - top {len(result.hits)}"
        if result.notes:
            status += "\n\n" + "\n".join(f"- {n}" for n in result.notes)
        return gallery_items(result.hits), ocr_rows(result.hits), status

    def pick_example(qid: str, arm: str, top_k: int):
        if not qid:
            return gr.update(), gr.update(), gr.update(), gr.update()
        text = by_qid[qid]
        return (text, *run(text, arm, top_k))

    with gr.Blocks(title="Multimodal search") as app:
        gr.Markdown(
            "# Multimodal search: CLIP vs OCR vs fusion\n"
            f"Gallery of {len(gallery.items)} images (COCO = visual, TextCaps = reading). "
            "Images load from their original hosts."
        )
        if notices:
            gr.Markdown("\n".join(f"> **Notice:** {n}" for n in notices))
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="Query", placeholder="a dog catching a frisbee", lines=1, autofocus=True
                )
                example = gr.Dropdown(
                    choices=choices,
                    value="",
                    label="Example captions (stored embeddings, no CLIP download)",
                )
            with gr.Column(scale=2):
                arm = gr.Dropdown(
                    choices=[(f"{a} - {ARM_HELP[a]}", a) for a in ARMS],
                    value="clip",
                    label="Arm",
                )
                top_k = gr.Slider(1, 24, value=DEFAULT_TOP_K, step=1, label="Top-k")
                go = gr.Button("Search", variant="primary")
        gr.Markdown(engine.weight_note())
        status = gr.Markdown()
        images = gr.Gallery(label="Results", columns=4, height="auto", object_fit="contain")
        ocr = gr.Dataframe(
            headers=["rank", "image_id", "OCR text"],
            datatype=["number", "str", "str"],
            label="OCR text of each hit",
            interactive=False,
            wrap=True,
        )
        example.change(pick_example, [example, arm, top_k], [query, images, ocr, status])
        go.click(run, [query, arm, top_k], [images, ocr, status])
        query.submit(run, [query, arm, top_k], [images, ocr, status])
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Gradio demo for the multimodal search.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "dev")
    args = parser.parse_args(argv)
    app = build_app(args.data_dir, args.results_dir)
    app.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
