"""The dev/test split: deterministic, by image, stratified by source."""

from __future__ import annotations

import random

import pytest

from mmsearch.gallery import GalleryItem
from mmsearch.split import split_images


def _items(n_coco: int = 500, n_text: int = 500) -> list[GalleryItem]:
    return [GalleryItem(f"coco-{i}", "coco") for i in range(n_coco)] + [
        GalleryItem(f"textcaps-{i}", "textcaps") for i in range(n_text)
    ]


def test_deterministic():
    assert split_images(_items()) == split_images(_items())


def test_independent_of_input_order():
    items = _items()
    shuffled = items[:]
    random.Random(1).shuffle(shuffled)
    assert split_images(items) == split_images(shuffled)


def test_every_image_in_exactly_one_split_and_stratified_counts():
    items = _items()
    assignment = split_images(items)
    assert set(assignment) == {i.image_id for i in items}
    assert set(assignment.values()) == {"dev", "test"}
    for source in ("coco", "textcaps"):
        dev = [i for i in items if i.source == source and assignment[i.image_id] == "dev"]
        assert len(dev) == 100  # exactly 20% of 500, not approximately


def test_seed_changes_the_assignment():
    a, b = split_images(_items(), seed=1), split_images(_items(), seed=2)
    assert a != b


def test_dev_fraction_is_respected_on_odd_sizes():
    assignment = split_images(_items(37, 11), dev_fraction=0.2)
    dev_coco = sum(1 for k, v in assignment.items() if k.startswith("coco") and v == "dev")
    dev_text = sum(1 for k, v in assignment.items() if k.startswith("textcaps") and v == "dev")
    assert (dev_coco, dev_text) == (round(0.2 * 37), round(0.2 * 11))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="dev_fraction"):
        split_images(_items(5, 5), dev_fraction=fraction)
