"""Deterministic dev/test split by image.

The rule, reproducible from code alone (nothing about the split is committed):

1. Group images by ``source`` (stratification).
2. Within a source, sort images by ``sha256(f"{seed}:{image_id}")``.
3. The first ``round(dev_fraction * n)`` images of each source are **dev**, the rest **test**.

Sorting by a hash makes the assignment independent of the order images appear in the file, and
taking a prefix of the sorted list makes the dev share exact per source (100 of 500 at 0.2)
instead of only approximately 20%. All captions of an image go with it: splitting is by image,
never by query.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from mmsearch.gallery import GalleryItem

DEFAULT_DEV_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 47


def _key(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()


def split_images(
    items: Sequence[GalleryItem],
    *,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, str]:
    """Map every ``image_id`` to ``"dev"`` or ``"test"``."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must be strictly between 0 and 1, got {dev_fraction}")
    by_source: dict[str, list[str]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item.image_id)

    assignment: dict[str, str] = {}
    for ids in by_source.values():
        ordered = sorted(ids, key=lambda image_id: _key(seed, image_id))
        n_dev = round(dev_fraction * len(ordered))
        for rank, image_id in enumerate(ordered):
            assignment[image_id] = "dev" if rank < n_dev else "test"
    return assignment
