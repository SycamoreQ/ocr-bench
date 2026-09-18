"""Contract tests: every adapter must obey the OCRAdapter interface.

These do NOT check recognition accuracy (that's what the IAM benchmark
run itself is for) — they just check that setup()/recognize()/teardown()
behave the way runner.py assumes they do, using one tiny synthetic image
generated on the fly (no IAM download needed to run this file).

Usage:

    pytest tests/test_adapters.py -k tesseract -v
    pytest tests/test_adapters.py -v                    # everything
    pytest tests/test_adapters.py -v -m "not slow"       # skip the VLM ones

An adapter whose is_available() returns False is skipped (not failed) —
that's expected for engines you haven't installed / aren't on this OS.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ocr_benchmark.adapters import ADAPTER_REGISTRY
from ocr_benchmark.adapters.base import AdapterUnavailableError

SLOW_ADAPTERS = {"granite-docling", "paddleocr-vl", "surya"}


@pytest.fixture(scope="session")
def sample_image(tmp_path_factory) -> Path:
    """A trivial white-background, black-text image. Good enough to prove
    an adapter's plumbing works end-to-end; not a substitute for real
    accuracy testing against IAM."""
    path = tmp_path_factory.mktemp("ocr_fixtures") / "hello.png"
    img = Image.new("RGB", (300, 80), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 25), "hello world", fill="black")
    img.save(path)
    return path


@pytest.mark.parametrize("adapter_name", sorted(ADAPTER_REGISTRY))
def test_adapter_contract(adapter_name: str, sample_image: Path, request):
    if adapter_name in SLOW_ADAPTERS:
        request.node.add_marker(pytest.mark.slow)

    adapter_cls = ADAPTER_REGISTRY[adapter_name]
    adapter = adapter_cls()

    # is_available() must be cheap and must not raise.
    available = adapter.is_available()
    assert isinstance(available, bool)
    if not available:
        pytest.skip(f"{adapter_name} not available in this environment")

    try:
        adapter.setup()
    except AdapterUnavailableError as exc:
        pytest.skip(f"{adapter_name} declined setup: {exc}")

    try:
        text = adapter.recognize(sample_image)
        assert isinstance(text, str)
    finally:
        adapter.teardown()  # must not raise even if recognize() did
