"""The adapter contract.

Every OCR engine gets wrapped behind this one interface so the benchmark
runner never needs to know anything engine-specific. Three methods matter:

    setup()      - called once, before any images are processed. Load
                   models / spin up sessions / import heavy deps here,
                   NOT at module import time (otherwise importing the
                   adapters package pulls in every ML framework at once).
    recognize()  - called once per image. Must return plain text.
    teardown()   - called once, after the adapter's run is complete.
                   Free GPU memory, close processes, etc.

`is_available()` is a cheap pre-flight check (deps importable? right OS?)
so the runner can skip an adapter with a clear reason instead of crashing
the whole benchmark on one missing package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AdapterUnavailableError(RuntimeError):
    """Raised by setup() when the engine can't run in this environment
    (missing dependency, wrong OS, no model weights, etc). The runner
    catches this, records it, and moves on to the next adapter."""


class OCRAdapter(ABC):
    #: short machine-friendly identifier, e.g. "tesseract". Set as a class
    #: attribute on every subclass; used as the "adapter" column in the CSV.
    name: str = "unnamed-adapter"

    def is_available(self) -> bool:
        """Cheap check, no model loading. Override to check `import`s
        or platform (e.g. OcrMacAdapter should return False off macOS).
        Default assumes availability and lets setup() fail loudly instead."""
        return True

    def setup(self) -> None:
        """One-time, possibly-expensive initialization. Default no-op."""

    @abstractmethod
    def recognize(self, image_path: Path) -> str:
        """Return the engine's raw text transcription of a single image.

        Must not raise for "no text found" (return "" instead). It's fine
        to raise for genuine engine failures — the runner records the
        exception per-sample and keeps going.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """One-time cleanup after all samples are processed. Default no-op."""
