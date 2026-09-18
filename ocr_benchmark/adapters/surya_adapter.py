"""Surya, called directly (not a docling `OcrOptions` engine).

Surya's Python API has changed across versions — pre-0.17 builds construct
`RecognitionPredictor()`/`DetectionPredictor()` with no arguments; 0.17+
requires a shared `FoundationPredictor`. setup() detects which generation
is installed by trying the modern construction first and falling back on
TypeError, rather than pinning to one version.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter

_TAG_RE = re.compile(r"<[^>]+>")


class SuryaAdapter(OCRAdapter):
    name = "surya"

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._detection_predictor = None
        self._recognition_predictor = None

    def is_available(self) -> bool:
        try:
            import surya.detection  # noqa: F401
            import surya.recognition  # noqa: F401
        except ImportError:
            return False
        return True

    def setup(self) -> None:
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise AdapterUnavailableError(
                "surya-ocr is not installed. `pip install surya-ocr`."
            ) from exc

        try:
            # Modern (>=0.17) API: needs a shared FoundationPredictor.
            from surya.foundation import FoundationPredictor

            foundation = FoundationPredictor()
            self._recognition_predictor = RecognitionPredictor(
                foundation_predictor=foundation
            )
            self._detection_predictor = DetectionPredictor()
        except (ImportError, TypeError):
            # Legacy (<0.17) API: predictors are self-contained.
            self._recognition_predictor = RecognitionPredictor()
            self._detection_predictor = DetectionPredictor()

    def recognize(self, image_path: Path) -> str:
        assert self._recognition_predictor is not None, "call setup() first"
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        predictions = self._call_recognition_predictor(image)
        return self._extract_text(predictions)

    def _call_recognition_predictor(self, image):
        # Different surya generations expect different call signatures.
        # Try newest-first, fall back on TypeError rather than branching on
        # a version string we'd have to keep updating.
        attempts = (
            lambda: self._recognition_predictor([image]),
            lambda: self._recognition_predictor([image], self._detection_predictor),
            lambda: self._recognition_predictor(
                [image], [[self.lang]], self._detection_predictor
            ),
        )
        last_exc: Exception | None = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            "Could not call surya's RecognitionPredictor with any known "
            "signature — the installed surya-ocr version may use a newer "
            "API than this adapter expects."
        ) from last_exc

    @staticmethod
    def _extract_text(predictions) -> str:
        if not predictions:
            return ""
        result = predictions[0]

        # Line-based result (most surya versions): join each line's text.
        text_lines = getattr(result, "text_lines", None)
        if text_lines:
            return "\n".join(
                getattr(line, "text", "") for line in text_lines
            ).strip()

        # Some newer builds return a single HTML-ish `.text`/`.html` blob
        # with block markup; strip tags to get plain text.
        blob = getattr(result, "html", None) or getattr(result, "text", None)
        if blob:
            return _TAG_RE.sub(" ", blob).strip()

        return ""
