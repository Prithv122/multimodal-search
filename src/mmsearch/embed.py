"""Encoders: the protocol, a deterministic fake, and a loader for pluggable real ones.

No module in this package imports torch. The real CLIP adapter lives in a local script and is
handed in through :func:`load_encoder` (``"package.module:factory"``), so the test suite and CI
run on :class:`FakeEmbedder` and never download weights.

``FakeEmbedder`` links its two spaces on purpose. Every word has a fixed pseudo-random vector
(a hash of the word). A text embeds to the mean of its word vectors. An image is described by
the words of what it *visibly shows* (:class:`FakeImage`) and embeds to the mean of those same
word vectors plus a little per-image noise. A caption mentioning a visible word therefore lands
near its image, while a word that appears only as printed text in the image (and so is not in
``FakeImage.words``) carries no signal: exactly the "CLIP sees content and cannot read" profile
that the planted fixture needs.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mmsearch._vendor.dense import Embedder, l2_normalise

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class MultimodalEmbedder(Embedder, Protocol):
    """A text encoder that can also encode images into the same space.

    ``Embedder`` (vendored) already supplies ``name``, ``dim``, ``encode_documents(texts)`` and
    ``encode_query(text)``. Images are whatever the adapter understands (paths, arrays, ...).
    """

    def encode_images(self, images: Sequence[Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class FakeImage:
    """A stand-in image: a stable key plus the words of what it visibly shows."""

    key: str
    words: tuple[str, ...]


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def _gaussian(text: str, dim: int) -> np.ndarray:
    return np.random.default_rng(_seed(text)).standard_normal(dim)


class FakeEmbedder:
    """Deterministic, torch-free, with linked image and text spaces (see module docstring)."""

    def __init__(self, dim: int = 64, *, noise: float = 0.6, name: str = "fake-clip") -> None:
        self._dim = dim
        self._noise = noise
        self._name = name
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def _word(self, word: str) -> np.ndarray:
        vec = self._cache.get(word)
        if vec is None:
            vec = _gaussian("word:" + word, self._dim)
            self._cache[word] = vec
        return vec

    def _bag(self, words: Sequence[str]) -> np.ndarray:
        if not words:
            return np.zeros(self._dim)
        return np.mean([self._word(w) for w in words], axis=0)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._bag(_WORD.findall(t.lower())) for t in texts]
        return self._finish(rows)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]

    def encode_images(self, images: Sequence[FakeImage]) -> np.ndarray:
        rows = []
        for image in images:
            signal = l2_normalise(self._bag(image.words))
            noise = l2_normalise(_gaussian("image-noise:" + image.key, self._dim))
            rows.append(signal + self._noise * noise)
        return self._finish(rows)

    def _finish(self, rows: Sequence[np.ndarray]) -> np.ndarray:
        if not rows:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalise(np.asarray(rows, dtype=np.float64)).astype(np.float32)


def load_encoder(spec: str, *, fallback: Embedder | None = None) -> Embedder:
    """Resolve an encoder spec.

    ``"fake"`` gives a :class:`FakeEmbedder`. ``"package.module:attr"`` imports ``attr``; when
    it is callable (a class or factory) it is called with no arguments, otherwise it is used as
    the encoder object itself. The object only needs ``encode_query(text) -> vector`` for the
    ``search`` command.
    """
    if spec == "fake":
        return fallback if fallback is not None else FakeEmbedder()
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"encoder spec must be 'fake' or 'package.module:attr', got {spec!r}")
    target = getattr(importlib.import_module(module_name), attr)
    encoder = target() if callable(target) else target
    if not hasattr(encoder, "encode_query"):
        raise TypeError(f"{spec!r} did not produce an object with encode_query(text)")
    return encoder
