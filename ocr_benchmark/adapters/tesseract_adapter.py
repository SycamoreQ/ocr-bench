"""Reference implementation #1: Tesseract, routed through docling.

Read this one first — the other "worked" adapter (RapidOCR) follows the
exact same shape. The stub adapters (surya, granite-docling, paddleocr-vl,
unlimited-ocr) are yours to fill in against this pattern.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter
from .docling_common import build_converter


class TesseractAdapter(OCRAdapter):
    name = "tesseract"

    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self._converter = None

    def is_available(self) -> bool:
        # docling's TesseractOcrOptions shells out to the `tesseract` binary
        # (not just the Python package), so check for that first.
        return shutil.which("tesseract") is not None

    def setup(self) -> None:
        try:
            from docling.datamodel.pipeline_options import TesseractOcrOptions
        except ImportError as exc:
            raise AdapterUnavailableError(
                "docling is not installed. `pip install docling`."
            ) from exc

        if not self.is_available():
            raise AdapterUnavailableError(
                "tesseract binary not found on PATH. "
                "`sudo apt install tesseract-ocr` (or `brew install tesseract`)."
            )

        ocr_options = TesseractOcrOptions(lang=[self.lang])
        self._converter = build_converter(ocr_options)

    def recognize(self, image_path: Path) -> str:
        assert self._converter is not None, "call setup() before recognize()"
        result = self._converter.convert(str(image_path))
        return result.document.export_to_text().strip()
