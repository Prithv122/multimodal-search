"""The fake encoder's linked spaces, the encoder loader, and the no-torch rule."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from mmsearch.embed import FakeEmbedder, FakeImage, MultimodalEmbedder, load_encoder


def test_deterministic_across_instances():
    a, b = FakeEmbedder(32), FakeEmbedder(32)
    img = FakeImage("k", ("v001", "v002"))
    np.testing.assert_array_equal(a.encode_images([img]), b.encode_images([img]))
    np.testing.assert_array_equal(a.encode_query("v001 v002"), b.encode_query("v001 v002"))


def test_outputs_are_unit_length_float32():
    out = FakeEmbedder(32).encode_documents(["v001 v002", "v003"])
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_satisfies_protocol():
    assert isinstance(FakeEmbedder(8), MultimodalEmbedder)


def test_text_is_closer_to_the_image_that_shows_its_words():
    emb = FakeEmbedder(64)
    shows = FakeImage("a", ("v001", "v002", "v003", "v004"))
    other = FakeImage("b", ("v100", "v101", "v102", "v103"))
    images = emb.encode_images([shows, other])
    query = emb.encode_query("v002 v003")
    assert images[0] @ query > images[1] @ query + 0.2


def test_printed_only_words_carry_no_signal_to_the_image():
    emb = FakeEmbedder(256)
    shows = emb.encode_images([FakeImage("a", ("v001", "v002"))])[0]
    printed_only = emb.encode_query("r0001 r0002")
    assert abs(float(shows @ printed_only)) < 0.25


def test_load_encoder_specs():
    assert isinstance(load_encoder("fake"), FakeEmbedder)
    assert load_encoder("mmsearch.embed:FakeEmbedder").encode_query("v001").shape == (64,)
    with pytest.raises(ValueError, match="encoder spec"):
        load_encoder("nonsense")
    with pytest.raises(TypeError, match="encode_query"):
        load_encoder("mmsearch.embed:_WORD")


def test_package_never_imports_torch():
    code = (
        "import sys, mmsearch.cli, mmsearch.embed, mmsearch.evaluate, mmsearch.fixtures;"
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith('torch.')];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
