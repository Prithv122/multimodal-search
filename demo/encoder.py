"""CLIP text encoder for free-text queries, in the same space as ``data/embeddings/clip_img.npy``.

Needs the ``embed`` extra (torch and transformers); ``src/mmsearch`` itself never imports them.
It uses the checkpoint named in ``data/embeddings/meta.json`` and the same tokenisation as the
script that wrote the stored arrays (fast CLIP tokenizer, truncation at 77 tokens, L2
normalisation), so a caption encoded here matches its stored row.

Use it with the search command, from the repository root (``python -m`` puts the working
directory on ``sys.path``; the ``mmsearch`` console script does not)::

    uv run --extra embed python -m mmsearch.cli search "a dog catching a frisbee" \
        --arm clip --encoder demo.encoder:factory
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONTEXT_LENGTH = 77
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class ClipTextEncoder:
    """``encode_query(text)`` returns a unit-length float32 vector."""

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR, device: str = "cpu") -> None:
        import torch
        from transformers import CLIPModel, CLIPTokenizerFast

        meta = json.loads((Path(data_dir) / "embeddings" / "meta.json").read_text("utf-8"))
        checkpoint = meta["hf_checkpoint"]
        self._torch = torch
        self.device = device
        self.name = f"{meta.get('model_name', 'clip')}/{meta.get('pretrained', '')}"
        self.dim = int(meta["dim"])
        self._tokenizer = CLIPTokenizerFast.from_pretrained(checkpoint)
        self._model = CLIPModel.from_pretrained(checkpoint).eval().to(device)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        enc = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=CONTEXT_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._torch.inference_mode():
            out = self._model.get_text_features(**enc)
        feats = out if hasattr(out, "shape") else out.pooler_output  # transformers 5 output object
        vectors = feats.float().cpu().numpy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]


def factory() -> ClipTextEncoder:
    """No-argument entry point for ``mmsearch search --encoder demo.encoder:factory``."""
    return ClipTextEncoder()
